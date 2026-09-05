import os
from flask import Flask, jsonify, request
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. إنشاء تطبيق خادم Web باستخدام Flask
app = Flask(__name__)

# متغيرات عامة سيتم تحميل النموذج والـ Tokenizer فيها عند بدء الخادم
model = None
tokenizer = None

SYSTEM_PROMPT = (
    "You are a Socratic Python tutor for an interactive learning platform. "
    "When you see [GUIDE], give subtle hints only without full solution code. "
    "When you see [AUDIT], analyze the code for bugs, clean code, security, and PEP8 standards."
)


def load_socratic_model():
    """دالة تقوم بتحميل النموذج الأساسي ثم دمج أوزان الـ Adapter الخفيفة فوقه"""
    global model, tokenizer

    base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    # مسار مجلد الأوزان المخصصة (يمكن تغييره عبر متغير بيئة)
    adapter_path = os.environ.get("ADAPTER_PATH", "./qwen-socratic-adapter")

    print("⏳ جاري تحميل النموذج الأساسي والـ Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # إعدادات ضغط الذاكرة 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto"
    )

    print(f"⏳ جاري دمج أوزان الـ Adapter من المسار: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # وضع التقييم والتوليد (إيقاف تدريب الأوزان)
    print("✅ تم تحميل النموذج السقراطي بنجاح وهو جاهز لاستقبال الطلبات!")


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """نقطة نهاية الـ API لاستقبال أسئلة الطلاب وإرجاع رد المدرس السقراطي"""
    # قراءة بيانات الـ JSON القادمة من العميل
    data = request.get_json()

    # التحقق من وجود الحقول المطلوبة
    if not data or "message" not in data or "type" not in data:
        return (
            jsonify({"error": "يرجى تزويد الحقول المطلوبة: type و message"}),
            400,
        )

    request_type = data["type"]  # [GUIDE] أو [AUDIT]
    user_message = data["message"]

    # دمج الوسم مع رسالة المستخدم
    full_prompt = f"{request_type} {user_message}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]

    # صياغة النص بـ ChatML
    formatted_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

    # توليد الإجابة مع إيقاف حساب التدرجات لتوفير الذاكرة وسرعة الرد
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # استخراج النص المولد فقط
    generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
    response_text = tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    return jsonify({
        "status": "success",
        "type": request_type,
        "response": response_text,
    })


if __name__ == "__main__":
    # تشغيل التحميل ثم بدء الخادم على المنفذ 5000
    load_socratic_model()
    app.run(host="0.0.0.0", port=5000, debug=False)