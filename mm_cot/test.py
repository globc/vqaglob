from app import MMCOTRationale
from PIL import Image

if __name__ == "__main__":
    gen = MMCOTRationale()
    cool = gen.run("/vqaglob/Multimodal-CoT/api/61.png", "Question: What objects are displayed?\nSolution:")
    print(cool)