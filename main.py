import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from functions.get_files_info import schema_get_files_info
from functions.get_file_content import schema_get_file_content
from functions.run_python_file import schema_run_python_file
from functions.write_file import schema_write_file
from call_function import call_function


def main():

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    system_prompt = """
You are a helpful AI assistant.

When a user asks a question or makes a request, make a function call plan. 
You can perform the following operations:

- get_files_info: Use this ONLY when the user asks about a directory or folder path. 
- get_file_content: Use this ONLY when the user asks about the contents of a file path. 
- write_file: Use this to create or update files. 
- run_python_file: Use this to run Python files with optional arguments. 

All paths are relative to the working directory. 
"""

    if len(sys.argv) < 2:
        print("I need a prompt!")
        sys.exit(1)
    verbose_flag = False
    if len(sys.argv) == 3 and sys.argv[2] == "--verbose":
        verbose_flag = True
    prompt = sys.argv[1]

    messages = [
        types.Content(role="user", parts=[types.Part(text=prompt)]),
    ]

    available_functions = types.Tool(
        function_declarations=[
            schema_get_files_info,
            schema_get_file_content,
            schema_run_python_file,
            schema_write_file,
        ]
    )

    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=system_prompt
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=config,
    )

    if response is None or response.usage_metadata is None:
        print("response is malformed.")
        return
    if verbose_flag:
        print(f"User prompt: {prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(
            f"Response tokens: {response.usage_metadata.candidates_token_count}")

    if response.function_calls:
        for function_call_part in response.function_calls:
            result = call_function(function_call_part, verbose_flag)
            print(result)
    else:
        print(response.text)


main()
