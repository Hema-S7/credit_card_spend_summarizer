import json
import requests
from requests.exceptions import RequestException
import streamlit as ui
import uuid

# ==========================================================
# Configuration
# ==========================================================

SERVER_URL = "http://127.0.0.1:8000"

UPLOAD_ENDPOINT = f"{SERVER_URL}/api/v1/upload"
QUERY_ENDPOINT = f"{SERVER_URL}/api/v1/query"


# ==========================================================
# Streamlit Configuration
# ==========================================================

ui.set_page_config(
    page_title="Credit Card Spend Summarizer",
    page_icon="💳",
    layout="wide",
)


# ==========================================================
# Styling
# ==========================================================

ui.markdown(
    """
    <style>

    .block-container {
        padding-top: 3rem;
    }

    .metadata-label {
        color:#6b7280;
        font-size:0.82rem;
        font-weight:500;
        margin-top:10px;
        margin-bottom:3px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Header
# ==========================================================

ui.subheader("NORTHSTAR BANK")
ui.subheader("💳 Credit Card Spend Summarizer")

ui.caption("AI-powered monthly analysis of your credit card transactions.")


# ==========================================================
# Session State
# ==========================================================

if "session_id" not in ui.session_state:

    ui.session_state.session_id = str(uuid.uuid4())


if "chat_history" not in ui.session_state:

    ui.session_state.chat_history = []


# ==========================================================
# Upload Document
# ==========================================================


def upload_document(document):

    try:

        files = {
            "file": (
                document.name,
                document,
                "application/pdf",
            )
        }

        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            timeout=120,
        )

        return response

    except RequestException:

        return None


# ==========================================================
# Send Query
# ==========================================================


def send_user_query(query):

    payload = {
        "query": query,
        "session_id": ui.session_state.session_id,
    }

    try:

        response = requests.post(
            QUERY_ENDPOINT,
            json=payload,
            timeout=120,
            stream=True,
            headers={"Accept": "application/json, text/event-stream"},
        )

        return response

    except RequestException:

        return None

# ==========================================================
# Read SSE Stream
# ==========================================================


def read_sse_stream(response):

    complete_stream = ""

    try:

        for line in response.iter_lines(decode_unicode=True):

            if not line:

                continue

            if not line.startswith("data:"):

                continue

            data = line[len("data:") :].strip()

            if data == "[DONE]":

                break

            try:

                event = json.loads(data)

                token = event.get("token", "")

                if token:

                    complete_stream += token

            except json.JSONDecodeError:

                continue

    except Exception as error:

        ui.error(f"Error reading stream: {error}")

    return complete_stream


# ==========================================================
# Extract JSON Objects
# ==========================================================


def extract_json_objects(text):

    objects = []

    decoder = json.JSONDecoder()

    position = 0

    while position < len(text):

        start = text.find("{", position)

        if start == -1:

            break

        try:

            obj, end = decoder.raw_decode(text[start:])

            objects.append(obj)

            position = start + end

        except json.JSONDecodeError:

            position = start + 1

    return objects


# ==========================================================
# Extract Agent Response
# ==========================================================


def extract_agent_response(raw_response):

    default_metadata = {
        "query": "",
        "document_name": "",
        "page_no": "",
        "policy_citations": "",
        "sql_query_executed": None,
    }

    if not raw_response:

        return ("No response generated.", default_metadata)

    objects = extract_json_objects(raw_response)

    query_response = None

    for obj in objects:

        if isinstance(obj, dict) and "response" in obj:

            query_response = obj

    if query_response is None:

        return (
            "I couldn't generate a response right now. Please try again",
            default_metadata,
        )

    answer = query_response.get("response", "No response generated.")

    metadata = {
        "query": query_response.get("query", ""),
        "document_name": query_response.get("document_name", ""),
        "page_no": query_response.get("page_no", ""),
        "policy_citations": query_response.get("policy_citations", ""),
        "sql_query_executed": query_response.get("sql_query_executed", None),
    }

    return answer, metadata
# ==========================================================
# Display Metadata
# ==========================================================


def display_metadata(metadata):

    if not metadata:
        return

    document_name = metadata.get("document_name")

    page_no = metadata.get("page_no")

    policy_citations = metadata.get("policy_citations")

    visible_fields = []

    if document_name and document_name != "N/A":

        visible_fields.append(
            (
                "Document",
                document_name,
                "text",
            )
        )

    if page_no and page_no != "N/A":

        visible_fields.append(
            (
                "Page",
                page_no,
                "text",
            )
        )

    if policy_citations and policy_citations != "N/A":

        visible_fields.append(
            (
                "Policy Citations",
                policy_citations,
                "text",
            )
        )

    # No SQL Query added here

    if not visible_fields:

        return

    with ui.expander(
        "📋 Response Details",
        expanded=False,
    ):

        for label, value, field_type in visible_fields:

            ui.markdown(
                f"""
                <div class="metadata-label">
                {label}
                </div>
                """,
                unsafe_allow_html=True,
            )

            ui.write(value)


# ==========================================================
# Sidebar
# ==========================================================


with ui.sidebar:

    ui.header("📁 Knowledge Base")

    uploaded_file = ui.file_uploader("Upload Document", type=["pdf"])

    if uploaded_file:

        ui.write(f"Selected: {uploaded_file.name}")

        if ui.button("Upload Document", use_container_width=True):

            with ui.spinner("Uploading..."):

                result = upload_document(uploaded_file)

                if result is None:

                    ui.error("Unable to connect to server.")

                elif result.ok:

                    ui.success("Document uploaded successfully.")

                else:

                    ui.error(result.text)

    ui.divider()

    if ui.button("🗑 Clear Chat", use_container_width=True):

        ui.session_state.chat_history = []

        ui.session_state.session_id = str(uuid.uuid4())

        ui.rerun()


# ==========================================================
# Display Chat History
# ==========================================================


for message in ui.session_state.chat_history:

    with ui.chat_message(message["role"]):

        ui.markdown(message["message"])

        if message["role"] == "assistant" and message.get("metadata"):

            display_metadata(message["metadata"])


# ==========================================================
# Chat Input
# ==========================================================


query = ui.chat_input("Ask about your credit card related query...")


# ==========================================================
# Process Query
# ==========================================================


if query:

    ui.session_state.chat_history.append({"role": "user", "message": query})

    with ui.chat_message("user"):

        ui.markdown(query)

    with ui.chat_message("assistant"):

        placeholder = ui.empty()

        with ui.spinner("Running..."):

            response = send_user_query(query)

            if response is None:

                answer = "Unable to connect to server."

                metadata = {}

            elif response.ok:

                content_type = response.headers.get("content-type", "").lower()

                # ======================================
                # JSON Response
                # ======================================

                if "application/json" in content_type:

                    data = response.json()

                    answer = data.get("response", "No response generated.")

                    metadata = {
                        "query": data.get("query", ""),
                        "document_name": data.get("document_name", ""),
                        "page_no": data.get("page_no", ""),
                        "policy_citations": data.get("policy_citations", ""),
                    }

                # ======================================
                # SSE Response
                # ======================================

                elif "text/event-stream" in content_type:

                    raw_response = read_sse_stream(response)

                    answer, metadata = extract_agent_response(raw_response)

                else:

                    answer = "Unsupported response format."

                    metadata = {}

            else:

                try:
                        error_data = response.json()

                        detail = error_data.get("detail", {})

                        if isinstance(detail, dict):
                            answer = detail.get(
                                "message",
                                "Request blocked by guardrail."
                            )
                        else:
                            answer = str(detail)

                except Exception:
                        answer = response.text or (
                            "I couldn't process your request right now. "
                            "Please try again."
                        )

                metadata = {}

        placeholder.markdown(answer)

        display_metadata(metadata)

        ui.session_state.chat_history.append(
            {"role": "assistant", "message": answer, "metadata": metadata}
        )
