from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from PIL import Image
from Core.ModelUse import predict_image

# Create your views here.
def index(request):
    return render(request, "detectApp/index.html")

def detectimg(request):
    if request.method == "POST" and request.FILES.get("frame"):
        try:
            # 直接从 request.FILES 读取并转成 PIL Image
            image = Image.open(request.FILES["frame"]).convert("RGB")

            # 模型预测
            predicted_class = predict_image(image)

            return JsonResponse({"status": "success", "res": predicted_class})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)
