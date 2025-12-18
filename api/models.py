from typing import Optional
from fastapi import Form
from pydantic import BaseModel


DEFAULT_SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image to do requirements:
   - Recognize and Extract all visible text in the image as accurately as possible without any additional explanations or comments.
   - Ensure that the extracted text is organized and presented in a structured Markdown format.
   - Pay close attention to maintaining the original hierarchy and formatting, includeing any headings, subheadings, lists, tables or inline text.
   - If any text elements are ambiguous or partially readable, include them with appropriate notes or markers, such as [illegible].
   - Preserve the spatial relationships where applicable by mimicking the document layout in Markdown.
   - Don't omit any part of the page including headers, footers, tables, and subtext.
   Provide only the transcription without any additional comments."""


class OCRRequest(BaseModel):
    prompt: str = "What is in this image?"
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def as_form(
        cls,
        prompt: str = Form("What is in this image?"),
        system_prompt: Optional[str] = Form(
        "You are an advanced OCR assistant capable of reading dense text and multimodal documents. "
        "Your output must be strictly structured Markdown. "
        "Rules:\n"
        "1. Use proper headings (##) for structure.\n"
        "2. Ensure that the extracted text is organized and presented in a structured Markdown format.\n"
        "3. Pay close attention to maintaining the original hierarchy and formatting, includeing any headings, subheadings, lists, tables or inline text.\n"
        "4. If any text elements are ambiguous or partially readable, include them with appropriate notes or markers, such as [illegible].\n"
        "5. Preserve the spatial relationships where applicable by mimicking the document layout in Markdown.\n"
        "6. Don't omit any part of the page including headers, footers, tables, and subtext.\n"
        "7. Enclose ALL code snippets in correct triple-backtick code blocks with the language specified (e.g., ```python, ```json, ```bash).\n"
        "8. Format all tables using Markdown table syntax.\n"
        "9. Use LaTeX for math formulas, enclosed in $...$ for inline and $$...$$ for block equations.\n"
        "10. Do not output conversational filler."
    ),
    ):
        return cls(prompt=prompt, system_prompt=system_prompt)


class OCRResponse(BaseModel):
    filename: str
    prompt: str
    system_prompt: Optional[str]
    response: str
