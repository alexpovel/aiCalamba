use axum::{
    extract::{Json, Multipart},
    response::{Html, IntoResponse},
    routing::{get, post},
    Form, Router,
};
use base64::prelude::*;
use log::{debug, info};
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{self, ChatCompletionRequest, ImageUrl},
    common::GPT4_O,
};
use serde::{Deserialize, Serialize};
use std::{env, error::Error};
use thirtyfour::prelude::*;
use tracing_subscriber::field::debug;
use url::Url;

#[derive(Deserialize, Debug)]
struct TextRequest {
    text: String,
}

#[derive(Deserialize, Debug)]
struct ImageRequest {
    /// base64
    image: String,
}

#[derive(Serialize, Debug)]
struct IcsResponse {
    ics: String,
}

async fn index() -> impl IntoResponse {
    Html(include_str!("index.html"))
}

async fn handle_text(Form(payload): Form<TextRequest>) -> impl IntoResponse {
    debug!("Handling text input: {payload:?}");

    match Url::parse(&payload.text) {
        Ok(url) => {
            debug!("Text input is URL: {url}");

            match process_url(&url).await {
                Ok(ics) => ics.into_response(),
                Err(e) => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Error: {}", e),
                )
                    .into_response(),
            }
        }
        Err(_) => {
            debug!("Text input is raw text.");

            match process_text(payload.text).await {
                Ok(ics) => ics.into_response(),
                Err(e) => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Error: {}", e),
                )
                    .into_response(),
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
            Err(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error: {}", e),
            )
                .into_response(),
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

enum LlmInput {
    Image(String),
    Text(String),
}

async fn fetch_llm_hallucinations(input: LlmInput) -> Result<String, Box<dyn Error + Send + Sync>> {
    let client = OpenAIClient::new(env::var("OPENAI_KEY")?);
    let content = match input {
                LlmInput::Image(img) => {
                    chat_completion::Content::ImageUrl(Vec::from(
                    [
                    ImageUrl {
                        r#type: chat_completion::ContentType::text,
                        text: Some(String::from("The following is a picture containing information for an event. Extract the information and format it in text format according to the iCal specification. Return nothing but that text.")),
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
                }
                LlmInput::Text(text) => chat_completion::Content::Text(format!("The following is the textual description of an event. Extract the information and format it in text format according to the iCal specification. Return nothing but that text.\n\n{text}")),
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

    let result = client.chat_completion(req).await?;
    let content = result.choices[0]
        .message
        .content
        .clone()
        .ok_or("No LLM response")?;

    Ok(content)
}

async fn fetch_screenshot(url: &Url) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut caps = DesiredCapabilities::chrome();
    caps.add_arg("--headless=new")?;
    let driver = WebDriver::new("http://localhost:9123", caps).await?;
    driver.goto(url.as_str()).await?;

    // Wait
    let element = driver.find(By::Css("body")).await?;
    element.wait_until().displayed().await?;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let base64img = driver.screenshot_as_png_base64().await?;
    driver.quit().await?;

    Ok(base64img)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/text", post(handle_text))
        .route("/image", post(handle_image));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
