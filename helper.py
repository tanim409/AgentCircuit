import hashlib
import json
import time

from langchain_core.messages import AIMessage, HumanMessage


def invoked_structured_output(llm, schema, input_messages, max_retries):
    messages = list(input_messages)
    runnable = llm.with_structured_output(schema, include_raw=True)
    for attempt in range(1, max_retries + 1):
        response = runnable.invoke(messages)
        if response.get("parsed") is not None and not response.get("parsing_error"):
            return response["parsed"]

        raw_output = response.get("raw")
        error = response.get("parsing_error")

        if raw_output and error:
            bad_content = raw_output.content if isinstance(raw_output, AIMessage) else str(raw_output)

            messages.extend([
                AIMessage(content=bad_content),
                HumanMessage(content=(
                    f"Your previous response caused an execution error when validating against the target schema.\n"
                    f"Specific Error: {error}\n"
                    f"Please review your previous output, fix the reported issue, and return valid output matching the required schema."
                ))
            ])
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Failed to get valid structured output after {max_retries} attempts.")


def hash_tool_call(tool_name: str, tool_args: dict):
    tool_call = {
        'tool': tool_name,
        'args': tool_args
    }
    toolTostr = json.dumps(tool_call, sort_keys=True, default=str)
    return hashlib.sha256(toolTostr.encode("utf-8")).hexdigest()
