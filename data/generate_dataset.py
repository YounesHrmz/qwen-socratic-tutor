import json
import os

# 1. التوجيه الدائم لشخصية النموذج
SYSTEM_PROMPT = (
    "You are a Socratic Python tutor for an interactive learning platform. "
    "When you see [GUIDE], give subtle hints only without full solution code. "
    "When you see [AUDIT], analyze the code for bugs, clean code, security, and PEP8 standards."
)

# 2. القائمة الخام للأمثلة التدريبية
RAW_SAMPLES = [
    {
        "type": "[GUIDE]",
        "user_input": "عندي خطأ IndentationError بعد حلقة if",
        "bot_response": "هذا الخطأ يعني أن بايثون تتوقع إزاحة (4 مسافات) داخل جسم عبارة `if`. افحص السطر الذي يلي `if` مباشرة ورتب الإزاحة."
    },
    {
        "type": "[GUIDE]",
        "user_input": "كيف أطبع الأرقام من 1 إلى 5 باستخدام for loop؟",
        "bot_response": "فكر في استخدام دالة `range()`. تذكر أن القيمة الأخيرة تكون غير شاملة، فما القيمة المناسبة كحد أقصى للوصول للرقم 5؟"
    },
    {
        "type": "[AUDIT]",
        "user_input": "افحص الكود التالي:\ndef add(a, b):\n return a+b",
        "bot_response": "📋 **نتيجة الفحص (Code Audit):**\n- **الصحة:** الكود يعمل.\n- **PEP8:** ينقص مسافة بادئة 4 مسافات قبل `return` ومسافات حول المشغل `+` ليصبح `return a + b`."
    },
    {
        "type": "[AUDIT]",
        "user_input": "افحص كود القسمة:\ndef div(a, b):\n    return a / b",
        "bot_response": "📋 **نتيجة الفحص (Code Audit):**\n- **خطأ منطقي محتمل:** إذا كانت قيمة `b` تساوي 0، سينتهي البرنامج بـ `ZeroDivisionError`.\n- **التصحيح:** أضف فحصاً أو استخدم `try/except` لمعالجة القسمة على صفر."
    }
]

def build_jsonl_dataset(output_path: str) -> None:
    formatted_data = []

    # المرور على كل مثال وتشكيله بقالب ChatML
    for sample in RAW_SAMPLES:
        full_user_text = f"{sample['type']} {sample['user_input']}"
        chat_structure = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_text},
                {"role": "assistant", "content": sample["bot_response"]}
            ]
        }
        formatted_data.append(chat_structure)

    # حفظ البيانات بتنسيق JSONL وترميز UTF-8
    with open(output_path, "w", encoding="utf-8") as file_handle:
        for entry in formatted_data:
            json_string = json.dumps(entry, ensure_ascii=False)
            file_handle.write(json_string + "\n")

    print(f"✅ تم توليد ملف البيانات بنجاح في: {output_path}")

if __name__ == "__main__":
    # تحديد مسار حفظ الملف في المجلد الرئيسي للمشروع
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, "socratic_python_dataset.jsonl")
    build_jsonl_dataset(target_file)