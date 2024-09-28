use axum::{
    extract::{DefaultBodyLimit, Multipart},
    response::{Html, IntoResponse},
    routing::{get, post},
    Form, Router,
};
use base64::prelude::*;
use log::{debug, error, info};
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{self, ChatCompletionRequest, ImageUrl},
    common::GPT4_O,
};
use serde::Deserialize;
use std::{env, error::Error, fmt::Display};
use thirtyfour::prelude::*;
use url::Url;

#[derive(Deserialize, Debug)]
struct TextRequest {
    text: String,
}

type WhateverError = Box<dyn Error + Send + Sync>;

async fn index() -> impl IntoResponse {
    Html(include_str!("index.html"))
}

fn internal_error(e: WhateverError) -> impl IntoResponse {
    error!("Server ran into an error: {e}");

    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "Internal server error",
    )
}

async fn handle_text(Form(payload): Form<TextRequest>) -> impl IntoResponse {
    debug!("Handling text input: {payload:?}");

    match Url::parse(&payload.text) {
        Ok(url) => {
            debug!("Text input is URL: {url}");

            match process_url(&url).await {
                Ok(ics) => ics.into_response(),
                Err(e) => internal_error(e).into_response(),
            }
        }
        Err(_) => {
            debug!("Text input is raw text.");

            match process_text(payload.text).await {
                Ok(ics) => ics.into_response(),
                Err(e) => internal_error(e).into_response(),
            }
        }
    }
}

async fn handle_image(mut multipart: Multipart) -> impl IntoResponse {
    if let Some(field) = multipart.next_field().await.unwrap() {
        let data = field.bytes().await.unwrap();
        let base64_image = BASE64_STANDARD.encode(data);

        match fetch_llm_hallucinations(LlmInput::Image(base64_image)).await {
            Ok(ics) => ics.into_response(),
            Err(e) => internal_error(e).into_response(),
        }
    } else {
        (
            axum::http::StatusCode::BAD_REQUEST,
            "No image file found".to_string(),
        )
            .into_response()
    }
}

async fn process_text(text: String) -> Result<String, Box<dyn Error + Send + Sync>> {
    fetch_llm_hallucinations(LlmInput::Text(text)).await
}

async fn process_url(url: &Url) -> Result<String, Box<dyn Error + Send + Sync>> {
    let base64img = fetch_screenshot(url).await?;

    let content = fetch_llm_hallucinations(LlmInput::Image(base64img)).await?;

    Ok(content)
}

#[derive(Debug)]
enum LlmInput {
    /// Base64-encoded image text.
    Image(String),
    /// Raw text.
    Text(String),
}

impl Display for LlmInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmInput::Image(i) => write!(f, "Image({:.20})", i.escape_debug()),
            LlmInput::Text(t) => write!(f, "Text({})", t.escape_debug()),
        }
    }
}

/// Visits LLM API and fetches model response.
async fn fetch_llm_hallucinations(input: LlmInput) -> Result<String, Box<dyn Error + Send + Sync>> {
    debug!("Will fetch LLM hallucinations for input: {input}");

    let now = chrono::Utc::now();

    let client = OpenAIClient::new(env::var("OPENAI_KEY")?);
    let content = match input {
        LlmInput::Image(img) => {
            chat_completion::Content::ImageUrl(Vec::from(
                [
                    ImageUrl {
                        r#type: chat_completion::ContentType::text,
                        text: Some(format!("The following is a picture containing information for an event. Extract the information and format it in text format according to the iCal specification. If parts are missing, such as the current year or month, fill it out from the current timestamp, which is {now}. Return nothing but that text. The image is shown below.")),
                        image_url: None,
                    },
                    ImageUrl {
                        r#type: chat_completion::ContentType::image_url,
                        text: None,
                        image_url: Some(chat_completion::ImageUrlType {
                            url: format!("data:image/jpeg;base64,{img}"),
                        }),
                    },
                ],
            ))
        },
        LlmInput::Text(text) => chat_completion::Content::Text(format!("The following is the textual description of an event. Extract the information and format it in text format according to the iCal specification. Return nothing but that text. If parts are missing, such as the current year or month, fill it out from the current timestamp, which is {now}. The text is:\n\n{text}")),
    };

    let req = ChatCompletionRequest::new(
        GPT4_O.to_string(),
        Vec::from([chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::user,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }]),
    );
    debug!("LLM request: {req:?}");

    let result = client.chat_completion(req).await?;
    let content = result.choices[0]
        .message
        .content
        .clone()
        .ok_or("No LLM response")?;

    debug!("LLM hallucinations: {}", content.escape_debug());
    Ok(content)
}

async fn fetch_screenshot(url: &Url) -> Result<String, WhateverError> {
    debug!("Will fetch screenshot for URL: {url}");

    let mut caps = DesiredCapabilities::chrome();
    caps.add_arg("--headless=new")?;
    let driver = WebDriver::new(
        env::var("CHROME_DRIVER_URL").unwrap_or("http://localhost:9123".into()),
        caps,
    )
    .await?;
    debug!("Driver created");
    driver.goto(url.as_str()).await?;

    // Wait
    let element = driver.find(By::Css("body")).await?;
    element.wait_until().displayed().await?;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    debug!("Site loaded");

    let base64img = driver.screenshot_as_png_base64().await?;
    driver.quit().await?;
    debug!("Screenshot taken");

    Ok(base64img)
}

#[tokio::main]
async fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    info!("Starting server");

    let app = Router::new()
        .route("/", get(index))
        .route("/text", post(handle_text))
        .route("/image", post(handle_image))
        // Need to handle images larger than 2 MB (axum default)
        .layer(DefaultBodyLimit::max(10_000_000));

    info!("App built: {app:?}");

    let listener = tokio::net::TcpListener::bind(env::var("ADDR").unwrap_or("0.0.0.0:3000".into()))
        .await
        .unwrap();

    info!("Will listen on: {listener:?}");
    axum::serve(listener, app).await.unwrap();
}
