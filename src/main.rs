use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
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
use std::{env, error::Error, fmt::Display, sync::Arc};
use tokio::sync::RwLock;
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

async fn handle_text(
    State(state): State<Arc<AppState>>,
    Form(payload): Form<TextRequest>,
) -> impl IntoResponse {
    debug!("Handling text input: {payload:?}");

    let text = payload.text.trim();

    match Url::parse(text) {
        Ok(url) => {
            debug!("Text input is URL: {url}");

            match process_url(&url, state).await {
                Ok(ics) => ics.into_response(),
                Err(e) => internal_error(e).into_response(),
            }
        }
        Err(_) => {
            debug!("Text input is raw text.");

            match process_text(text).await {
                Ok(ics) => ics.into_response(),
                Err(e) => internal_error(e).into_response(),
            }
        }
    }
}

async fn handle_image(mut multipart: Multipart) -> impl IntoResponse {
    match multipart.next_field().await {
        Ok(Some(field)) => match field.bytes().await {
            Ok(img) => match fetch_llm_hallucinations(LlmInput::Image(&img)).await {
                Ok(ics) => ics.into_response(),
                Err(e) => internal_error(e).into_response(),
            },
            Err(e) => {
                error!("Failed to read image file: {e}");
                (
                    axum::http::StatusCode::BAD_REQUEST,
                    "Failed to read image file".to_string(),
                )
                    .into_response()
            }
        },
        Ok(None) => {
            error!("Failed to read image file");
            (
                axum::http::StatusCode::BAD_REQUEST,
                "Failed to read image file".to_string(),
            )
                .into_response()
        }
        Err(e) => {
            error!("Failed to read image file: {e}");
            (
                axum::http::StatusCode::BAD_REQUEST,
                "Failed to read image file".to_string(),
            )
                .into_response()
        }
    }
}

async fn process_text(text: &str) -> Result<String, WhateverError> {
    fetch_llm_hallucinations(LlmInput::Text(text)).await
}

async fn process_url(url: &Url, state: Arc<AppState>) -> Result<String, WhateverError> {
    let img = fetch_screenshot(url).await?;
    *state.last_image.write().await = Some(img.clone());
    let content = fetch_llm_hallucinations(LlmInput::Image(&img)).await?;
    Ok(content)
}

#[derive(Debug)]
enum LlmInput<'a> {
    /// Image.
    Image(&'a [u8]),
    /// Raw text.
    Text(&'a str),
}

impl Display for LlmInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmInput::Image(i) => write!(f, "Image({:?})", i.get(..20)),
            LlmInput::Text(t) => write!(f, "Text({})", t.escape_debug()),
        }
    }
}

/// Visits LLM API and fetches model response.
async fn fetch_llm_hallucinations(input: LlmInput<'_>) -> Result<String, WhateverError> {
    debug!("Will fetch LLM hallucinations for input: {input}");

    let now = chrono::Utc::now().format("%Y-%m-%d");

    let common_prompt = format!(
        r"Extract the information and format it in text format according to the iCal specification.
Return nothing but that text.
If date info is missing, such as the current year, month or day, fill it in from the current date, which is {now}.
If no wall clock time is mentioned, make it an all-day event.
Assume event times are in Europe/Berlin aka CEST timezone.
Pay attention to events spanning multiple days, and recurring events.
If only a start time is mentioned but no end time, assume one hour duration."
    );

    let client = OpenAIClient::new(env::var("OPENAI_KEY")?);
    let content = match input {
        LlmInput::Image(img) => {
            chat_completion::Content::ImageUrl(Vec::from(
                [
                    ImageUrl {
                        r#type: chat_completion::ContentType::text,
                        text: Some(format!("The following is a picture containing information for an event. {common_prompt}\nThe image is shown below.")),
                        image_url: None,
                    },
                    ImageUrl {
                        r#type: chat_completion::ContentType::image_url,
                        text: None,
                        image_url: Some(chat_completion::ImageUrlType {
                            url: format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(img)),
                        }),
                    },
                ],
            ))
        },
        LlmInput::Text(text) => chat_completion::Content::Text(format!("The following is the textual description of an event. {common_prompt}\nThe text is:\n\n{text}")),
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

    // Sanity check
    if let Err(e) = icalendar::parser::read_calendar(&content) {
        // Log...
        error!("Failed to parse iCal content: {e}");
        // ... and send it out anyway. Clients might tolerate it.
    } else {
        debug!("Parsed iCal content successfully");
    }

    Ok(content)
}

async fn fetch_screenshot(url: &Url) -> Result<Vec<u8>, WhateverError> {
    debug!("Will fetch screenshot for URL: {url}");

    // https://apiflash.com/documentation
    let params = [
        ("access_key", env::var("APIFLASH_KEY")?),
        ("url", url.to_string()),
        // Some pages are really slow. We're not in a rush, make sure results are correct.
        ("delay", "10".to_string()),
    ];

    let client = reqwest::Client::new();
    let request = client
        .request(
            reqwest::Method::GET,
            "https://api.apiflash.com/v1/urltoimage",
        )
        .query(&params)
        .build()?;
    debug!("Request: {request:?}");

    let response = client.execute(request).await?;
    debug!("Response: {response:?}");

    let bytes = response.bytes().await?;
    // Sanity check
    let mut decoder = jpeg_decoder::Decoder::new(bytes.as_ref());
    decoder.decode()?;
    debug!("Image size: {} bytes", bytes.len());

    Ok(bytes.into())
}

async fn serve_last_image(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.last_image.read().await.as_ref() {
        Some(img) => axum::response::Response::builder()
            .header("Content-Type", "image/jpeg")
            .body(axum::body::Body::from(img.clone()))
            .unwrap(),
        None => (
            axum::http::StatusCode::NOT_FOUND,
            "No image available".to_string(),
        )
            .into_response(),
    }
}

struct AppState {
    last_image: RwLock<Option<Vec<u8>>>,
}

#[tokio::main]
async fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    info!("Starting server");

    let state = Arc::new(AppState {
        last_image: RwLock::new(None),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/text", post(handle_text))
        .route("/image", post(handle_image))
        .route("/image/last", get(serve_last_image)) // For debugging
        // Need to handle images larger than 2 MB (axum default)
        .layer(DefaultBodyLimit::max(10_000_000))
        .with_state(state);

    info!("App built: {app:?}");

    let listener = tokio::net::TcpListener::bind(env::var("ADDR").unwrap_or("0.0.0.0:3000".into()))
        .await
        .unwrap();

    info!("Will listen on: {listener:?}");
    axum::serve(listener, app).await.unwrap();
}
