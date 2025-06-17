import re

def extract_and_sum_prompt_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 找出所有 TOTAL_TOKENS: 後面的數字
    token_counts = re.findall(r'PROMPT_TOKENS:\s*(\d+)', content)
    
    # 轉成整數並加總
    token_sum = sum(int(num) for num in token_counts)
    
    return token_sum

def extract_and_sum_completion_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 找出所有 TOTAL_TOKENS: 後面的數字
    token_counts = re.findall(r'COMPLETION_TOKENS:\s*(\d+)', content)
    
    # 轉成整數並加總
    token_sum = sum(int(num) for num in token_counts)
    
    return token_sum

def extract_and_sum_total_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 找出所有 TOTAL_TOKENS: 後面的數字
    token_counts = re.findall(r'TOTAL_TOKENS:\s*(\d+)', content)
    
    # 轉成整數並加總
    token_sum = sum(int(num) for num in token_counts)
    
    return token_sum

# 使用範例
file_path = 'results/prunetool_results/gpt4.1-gpt4.1-webqsp-ToG-log.txt'  # 替換為你的檔案路徑
total = extract_and_sum_prompt_tokens(file_path)
print(f"PROMPT_TOKENS 總和為: {total}")

total = extract_and_sum_completion_tokens(file_path)
print(f"COMPLETION_TOKENS 總和為: {total}")

total = extract_and_sum_total_tokens(file_path)
print(f"TOTAL_TOKENS 總和為: {total}")
