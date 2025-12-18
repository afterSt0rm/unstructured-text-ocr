import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pathlib import Path
from datetime import datetime

from api.models import OCRRequest, OCRResponse
from api.utils import (
    bytes_to_base64,
    extract_text_and_images_from_pdf,
    send_chat_completion_request,
)

app = FastAPI()


@app.get("/healthcheck")
async def healthcheck():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    request_data: OCRRequest = Depends(OCRRequest.as_form),
    file: UploadFile = File(...),
):
    """
    Endpoint to perform OCR on an uploaded image or PDF.
    Accepts an image/PDF file, text prompt and an optional system prompt.
    """
    try:
        # Read file contents
        contents = await file.read()
        
        response_texts = []

        if file.content_type == "application/pdf":
            # Extract Text + Embedded Images
            # Returns list of (text, list_of_image_bytes)
            pages = extract_text_and_images_from_pdf(contents)
            
            for i, (extracted_text, page_images) in enumerate(pages):
                images_base64 = []
                for img_bytes in page_images[:1]:
                     images_base64.append(bytes_to_base64(img_bytes, max_size=512))

                # Prompt: Digital Text + Figures
                hybrid_prompt = f"""Page {i+1} Content:
```
{extracted_text[:4000]}
```
I have also attached the {len(images_base64)} key figures/images found on this page.
Please transcribe the full page content in Markdown.
- Use the provided text trace as the source of truth for text.
- For attached images: Provide a comprehensive visual description of the chart/figure, explaining trends or content visible in the image.
- If the image is a logo or the same image is present on every page, do not include it in the response.
- Format code blocks and JSON strictly with correct syntax (```json, ```python).
- Do not hallucinate content not present in the text or images.
{request_data.prompt}"""
                
                page_response = send_chat_completion_request(
                    hybrid_prompt,
                    images_base64=images_base64,
                    system_prompt=request_data.system_prompt,
                )
                response_texts.append(f"--- Page {i+1} ---\n{page_response}")
        else:
            # Assume Image
            # Convert to base64
            image_base64 = bytes_to_base64(contents, max_size=512)

            # Send request to VLM
            response_text = send_chat_completion_request(
                request_data.prompt,
                images_base64=[image_base64], # Pass as list
                system_prompt=request_data.system_prompt,
            )
            response_texts.append(response_text)

        final_response = "\n\n".join(response_texts)
        
        # Auto-save to markdown file
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_output_{timestamp}.md"
        output_path = output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_response)
        
        print(f"✓ Saved output to: {output_path}")

        return OCRResponse(
            filename=file.filename,
            prompt=request_data.prompt,
            system_prompt=request_data.system_prompt,
            response=final_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
