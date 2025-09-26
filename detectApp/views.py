from django.shortcuts import render
from django.http import JsonResponse

from PIL import Image
from Core.ModelUse import predict_image

from .models import Disease
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

            disease=Disease.objects.get(name=predicted_class)

            return JsonResponse({"status": "success", "res": predicted_class,"solution":disease.solution,"harm":disease.harm})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
    return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)

def disease_detail(request, name):
    try:
        disease = Disease.objects.get(name=name)
    except Disease.DoesNotExist:
        return render(request, "detectApp/disease_not_found.html", {"name": name})

    return render(request, "detectApp/disease_detail.html", {"disease": disease})