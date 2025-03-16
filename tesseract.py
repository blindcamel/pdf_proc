import pytesseract
import numpy as np
import fitz
import cv2

file_path = "processed/platt1.pdf"
doc = fitz.open(str(file_path))
page_count = len(doc)


# def ocr_extract(doc):
#     text = ""
#     for page in doc:
#         pix = page.get_pixmap()
#         print(
#             f"Pixmap dims: {pix.width}x{pix.height}, samples per pixel: {pix.n}"
#         )  # Debug

#         img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
#             pix.height, pix.width, pix.n
#         )

#         print(f"Numpy array shape: {img.shape}")  # Debug

#         text += pytesseract.image_to_string(img) + "\n"
#         print(f"OCR output length: {len(text)}")  # Debug
#     doc.close()
#     return {"text": text.strip(), "source": "ocr", "page_count": page_count}



def ocr_extract_first_page(doc):
    text = ""
    if len(doc) > 0:
        page = doc[0]  # Get only the first page
        pix = page.get_pixmap()
        print(f"Pixmap dims: {pix.width}x{pix.height}, samples per pixel: {pix.n}")
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        print(f"Numpy array shape: {img.shape}")
        
        # Save the image as BMP
        bmp_path = "first_page.bmp"
        cv2.imwrite(bmp_path, img)
        print(f"Saved first page image to {bmp_path}")
        
        text = pytesseract.image_to_string(img)
        print(f"OCR output length: {len(text)}")
        print(text)
    
    doc.close()
    return text

# Call the function
ocr_extract_first_page(doc)
