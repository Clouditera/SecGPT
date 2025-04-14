import json
import re
import time
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

class CyberMetricEvaluator:
    def __init__(self, api_key, file_path, model="SecGPT", base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.file_path = file_path
        self.model = model

    def read_json_file(self):
        with open(self.file_path, 'r', encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    # def extract_answer(response):
    #     if response.strip():  
    #         match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
    #         if match:
    #             return match.group(1).upper()
    #     return None
    def extract_answer(response):
        if not response or not isinstance(response, str):
            return None

        response = response.strip()  # 清理前后空格

        # 1️⃣ **匹配 "ANSWER: X" 这种格式**
        match = re.search(r"ANSWER[:\s]*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()  # 确保返回大写 A/B/C/D

        # 2️⃣ **尝试从整个文本中提取单独的 A-D 作为答案**
        match = re.findall(r"\b([A-D])\b", response, re.IGNORECASE)
        if match:
            return match[0].upper()  # 返回第一个匹配的答案

        # 3️⃣ **支持 "A) "、"B. "、"C: "、"D、" 等常见格式**
        match = re.search(r"\b([A-D])[\)\.\:、]", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None  # 没有找到答案时返回 None

    def ask_llm(self, question, options, correct_answer,max_retries=5):
        options_str = ', '.join([f"{key}) {value}" for key, value in options.items()])
        prompt = f"Question: {question}\nOptions: {options_str}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert answering multiple-choice questions."},
                        {"role": "user", "content": prompt},
                    ]
                )
                if response.choices:
                    result = self.extract_answer(response.choices[0].message.content)
                    if result:
                        return result
                    else:
                        print("Incorrect answer format detected. Retrying...")
            except Exception as e:
                print(f"Error: {e}. Retrying in {2 ** attempt} seconds.")
                time.sleep(2 ** attempt)
        
        return None  

    def run_evaluation(self):
        json_data = self.read_json_file()
        questions_data = json_data

        correct_count = 0
        incorrect_answers = []
        category_errors = defaultdict(int)  
        category_total = defaultdict(int)   

        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            for item in questions_data:
                question = item['question']
                options = item['options']
                correct_answer = item['answer']
                category = item.get('category', 'Unknown')  

                category_total[category] += 1  

                llm_answer = self.ask_llm(question, options,correct_answer)
                if llm_answer == correct_answer:
                    correct_count += 1
                else:
                    incorrect_answers.append({
                        'question': question,
                        'options': options,
                        'category': category,
                        'correct_answer': correct_answer,
                        'llm_answer': llm_answer
                    })
                    category_errors[category] += 1  

                accuracy_rate = correct_count / (progress_bar.n + 1) * 100
                progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
                progress_bar.update(1)

        final_accuracy = correct_count / len(questions_data) * 100
        print(f"\nFinal Accuracy: {final_accuracy:.2f}%")

        # 计算每个类别的错误占比
        category_error_rate = {
            cat: (category_errors[cat] / category_total[cat]) * 100
            for cat in category_total
        }

        print("\nCategory-wise Error Distribution:")
        for category, error_rate in sorted(category_error_rate.items(), key=lambda x: -x[1]):
            print(f"  {category}: {error_rate:.2f}% ({category_errors[category]}/{category_total[category]})")

        if incorrect_answers:
            result_data = {
                "incorrect_answers": incorrect_answers,
                "category_error_distribution": category_error_rate
            }

            with open(f"{final_accuracy:.2f}%.json", "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)

# ======== 运行脚本 =========
if __name__ == "__main__":
    API_KEY = "your-api-key"  
    FILE_PATH = "./cissp-题目.json"  
    MODEL_NAME = "SecGPT-14B"  
    BASE_URL = "http://127.0.0.1:8003/v1"  

    evaluator = CyberMetricEvaluator(api_key=API_KEY, file_path=FILE_PATH, model=MODEL_NAME, base_url=BASE_URL)
    evaluator.run_evaluation()

