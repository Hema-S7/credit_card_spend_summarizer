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
# Custom Styling
# ==========================================================

ui.markdown(
    """
    <style>

        .block-container {
            padding-top: 3rem;
        }

        .metadata-label {
            color: #6b7280;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 10px;
            margin-bottom: 3px;
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

    except RequestException as error:

        ui.error(f"Upload failed: {error}")

        return None


# ==========================================================
# Send User Query
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
        )

        return response

    except RequestException as error:

        ui.error(f"Query failed: {error}")

        return None


# ==========================================================
# Read SSE Stream
# ==========================================================


def read_sse_stream(response):
    """
    Reads the streamed response from the backend.

    Expected format:

        data: {"token":"{"}
        data: {"token":"reason"}
        data: {"token":"..."}
        data: {"token":"}"}
        data: {"token":"{"}
        ...
        data: [DONE]

    The token stream is collected internally.
    Nothing is displayed directly.
    """

    complete_stream = ""

    try:

        for line in response.iter_lines(decode_unicode=True):

            if not line:
                continue

            # --------------------------------------------------
            # Only process SSE data events
            # --------------------------------------------------

            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()

            # --------------------------------------------------
            # End of stream
            # --------------------------------------------------

            if data == "[DONE]":
                break

            # --------------------------------------------------
            # Parse SSE event
            # --------------------------------------------------

            try:

                event = json.loads(data)

                token = event.get("token", "")

                if token:

                    complete_stream += token

            except json.JSONDecodeError:

                # Ignore invalid SSE JSON
                continue

    except Exception as error:

        ui.error(f"Error reading response stream: {error}")

    return complete_stream


# ==========================================================
# Extract JSON Objects
# ==========================================================


def extract_json_objects(text):
    """
    Extract multiple JSON objects from the reconstructed
    stream.

    Example:

        {"reason":"...","route":"VECTOR_DB"}
        {"document_name":"","page_no":"","response":"Hi"}

    Returns a list of JSON objects.
    """

    objects = []

    decoder = json.JSONDecoder()

    position = 0

    while position < len(text):

        # --------------------------------------------------
        # Find next JSON object
        # --------------------------------------------------

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
# Extract Final Agent Response
# ==========================================================


def extract_agent_response(raw_response):
    """
    Extracts the final QueryResponse.

    Intermediate response:

        {
            "reason": "...",
            "route": "VECTOR_DB"
        }

    Final response:

        {
            "document_name": "",
            "page_no": "",
            "policy_citations": "",
            "query": "hi",
            "response": "Hi! How can I help you today?",
            "sql_query_executed": null
        }
    """

    default_metadata = {
        "query": "",
        "document_name": "",
        "page_no": "",
        "policy_citations": "",
        "sql_query_executed": None,
    }

    if not raw_response:

        return (
            "No response generated.",
            default_metadata,
        )

    # ------------------------------------------------------
    # Extract all JSON objects
    # ------------------------------------------------------

    objects = extract_json_objects(raw_response)

    # ------------------------------------------------------
    # Find final QueryResponse
    # ------------------------------------------------------

    query_response = None

    for obj in objects:

        if isinstance(obj, dict) and "response" in obj:

            query_response = obj

    # ------------------------------------------------------
    # No final response found
    # ------------------------------------------------------

    if query_response is None:

        return (
            "Unable to parse server response.",
            default_metadata,
        )

    # ------------------------------------------------------
    # Main response
    # ------------------------------------------------------

    answer = query_response.get(
        "response",
        "No response generated.",
    )

    # ------------------------------------------------------
    # Metadata
    #
    # Query is retained internally but NOT displayed.
    # ------------------------------------------------------

    metadata = {
        "query": query_response.get(
            "query",
            "",
        ),
        "document_name": query_response.get(
            "document_name",
            "",
        ),
        "page_no": query_response.get(
            "page_no",
            "",
        ),
        "policy_citations": query_response.get(
            "policy_citations",
            "",
        ),
        "sql_query_executed": query_response.get(
            "sql_query_executed",
            None,
        ),
    }

    return answer, metadata


# ==========================================================
# Display Response Metadata
# ==========================================================


def display_metadata(metadata):

    if not metadata:
        return

    # ------------------------------------------------------
    # Get metadata
    #
    # Query is intentionally NOT displayed.
    # ------------------------------------------------------

    document_name = metadata.get("document_name")

    page_no = metadata.get("page_no")

    policy_citations = metadata.get("policy_citations")

    sql_query = metadata.get("sql_query_executed")

    # ------------------------------------------------------
    # Build only fields that contain values
    # ------------------------------------------------------

    visible_fields = []

    if document_name:

        visible_fields.append(
            (
                "Document",
                document_name,
                "text",
            )
        )

    if page_no:

        visible_fields.append(
            (
                "Page",
                page_no,
                "text",
            )
        )

    if policy_citations:

        visible_fields.append(
            (
                "Policy Citations",
                policy_citations,
                "text",
            )
        )

    if sql_query:

        visible_fields.append(
            (
                "SQL Query",
                sql_query,
                "sql",
            )
        )

    # ------------------------------------------------------
    # If nothing is available, don't show the dropdown
    # ------------------------------------------------------

    if not visible_fields:
        return

    # ------------------------------------------------------
    # Response Details Dropdown
    # ------------------------------------------------------

    with ui.expander(
        "📋 Response Details",
        expanded=False,
    ):

        for label, value, field_type in visible_fields:

            # ----------------------------------------------
            # Clean label
            # ----------------------------------------------

            ui.markdown(
                f'<div class="metadata-label">' f"{label}" f"</div>",
                unsafe_allow_html=True,
            )

            # ----------------------------------------------
            # SQL
            # ----------------------------------------------

            if field_type == "sql":

                ui.code(
                    value,
                    language="sql",
                )

            # ----------------------------------------------
            # Normal text
            # ----------------------------------------------

            else:

                ui.write(value)


# ==========================================================
# Sidebar
# ==========================================================

with ui.sidebar:

    ui.header("📁 Knowledge Base")

    uploaded_file = ui.file_uploader(
        "Upload Document",
        type=["pdf"],
    )

    if uploaded_file:

        ui.write(f"Selected: {uploaded_file.name}")

        if ui.button(
            "Upload Document",
            use_container_width=True,
        ):

            with ui.spinner("Uploading..."):

                result = upload_document(uploaded_file)

                if result is None:

                    ui.error("Unable to connect to server.")

                elif result.ok:

                    ui.success("Document uploaded successfully.")

                else:

                    ui.error(result.text)

    ui.divider()

    # ======================================================
    # Clear Chat
    # ======================================================

    if ui.button(
        "🗑 Clear Chat",
        use_container_width=True,
    ):

        ui.session_state.chat_history = []

        ui.session_state.session_id = str(uuid.uuid4())

        ui.rerun()


# ==========================================================
# Display Chat History
# ==========================================================

for message in ui.session_state.chat_history:

    with ui.chat_message(message["role"]):

        ui.markdown(message["message"])

        # --------------------------------------------------
        # Display metadata for assistant messages
        # --------------------------------------------------

        if message["role"] == "assistant" and message.get("metadata"):

            display_metadata(message["metadata"])


# ==========================================================
# Chat Input
# ==========================================================

query = ui.chat_input("Ask about your credit card related query...")


# ==========================================================
# Chat Processing
# ==========================================================

if query:

    # ======================================================
    # Save User Message
    # ======================================================

    ui.session_state.chat_history.append(
        {
            "role": "user",
            "message": query,
        }
    )

    # ======================================================
    # Display User Message
    # ======================================================

    with ui.chat_message("user"):

        ui.markdown(query)

    # ======================================================
    # Assistant Response
    # ======================================================

    with ui.chat_message("assistant"):

        placeholder = ui.empty()

        with ui.spinner("Generating insights..."):

            # ------------------------------------------------
            # Send query
            # ------------------------------------------------

            response = send_user_query(query)

            # ------------------------------------------------
            # Connection failure
            # ------------------------------------------------

            if response is None:

                answer = "Unable to connect to server."

                metadata = {}

            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------

            elif response.ok:

                # --------------------------------------------
                # Read stream silently
                # --------------------------------------------

                raw_response = read_sse_stream(response)

                # --------------------------------------------
                # Extract final response + metadata
                # --------------------------------------------

                answer, metadata = extract_agent_response(raw_response)

            # ------------------------------------------------
            # HTTP error
            # ------------------------------------------------

            else:

                answer = f"Request failed: " f"{response.status_code}"

                metadata = {}

                try:

                    error_text = response.text

                    if error_text:

                        ui.error(error_text)

                except Exception:

                    pass

        # ==================================================
        # Display Final Answer
        # ==================================================

        placeholder.markdown(answer)

        # ==================================================
        # Display Metadata
        # ==================================================

        display_metadata(metadata)

        # ==================================================
        # Save Assistant Message
        # ==================================================

        ui.session_state.chat_history.append(
            {
                "role": "assistant",
                "message": answer,
                "metadata": metadata,
            }
        )
