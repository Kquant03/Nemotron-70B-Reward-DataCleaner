import json
import requests
import tiktoken
from tqdm import tqdm

# Update this to your local endpoint URL
LOCAL_ENDPOINT = "http://localhost:8000/v1/chat/completions"

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def process_dataset(input_file, output_file, deleted_file):
    lines_kept = 0
    lines_removed = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile, open(deleted_file, 'w') as deletedfile:
        lines = infile.readlines()
        total_lines = len(lines)

        for line in tqdm(lines, total=total_lines, desc="Processing dataset"):
            data = json.loads(line)
            conversations = data['conversations']

            # Check token count
            total_tokens = sum(count_tokens(msg['value']) for msg in conversations)
            if total_tokens > 3800:
                data['score'] = 'Removed due to token count'
                deletedfile.write(json.dumps(data) + '\n')
                lines_removed += 1
                continue

            # Get reward model score
            system_message = next((msg for msg in conversations if msg['from'] == 'system'), None)
            human_messages = [msg for msg in conversations if msg['from'] == 'human']
            gpt_messages = [msg for msg in conversations if msg['from'] == 'gpt']

            messages = []
            for human_message, gpt_message in zip(human_messages, gpt_messages):
                messages.append({"role": "user", "content": human_message['value']})
                messages.append({"role": "assistant", "content": gpt_message['value']})

            if system_message:
                messages[-1]['content'] += f"\n\nMy thoughts: {system_message['value']}"

            if human_messages and gpt_messages:
                try:
                    response = requests.post(
                        LOCAL_ENDPOINT,
                        json={
                            "model": "nvidia/llama-3.1-nemotron-70b-reward",
                            "messages": messages
                        }
                    )
                    response.raise_for_status()
                    completion = response.json()
                    reward_score_str = completion['choices'][0]['message']['content']
                    reward_score = float(reward_score_str.split(':')[-1])

                    if reward_score >= -18.5:
                        cleaned_data = {
                            'conversations': [
                                {'from': 'system', 'value': system_message['value']} if system_message else None
                            ]
                        }
                        for human_message, gpt_message in zip(human_messages, gpt_messages):
                            cleaned_data['conversations'].append({'from': 'human', 'value': human_message['value']})
                            cleaned_data['conversations'].append({'from': 'gpt', 'value': gpt_message['value']})
                        cleaned_data['conversations'] = [conv for conv in cleaned_data['conversations'] if conv]
                        outfile.write(json.dumps(cleaned_data) + '\n')
                        lines_kept += 1
                    else:
                        data['score'] = reward_score
                        deletedfile.write(json.dumps(data) + '\n')
                        lines_removed += 1
                except requests.exceptions.RequestException as e:
                    print(f"Error calling local endpoint: {e}")
                    data['score'] = 'Error in API call'
                    deletedfile.write(json.dumps(data) + '\n')
                    lines_removed += 1
            else:
                data['score'] = 'Removed due to missing human or gpt message'
                deletedfile.write(json.dumps(data) + '\n')
                lines_removed += 1

    print(f"Lines kept: {lines_kept}")
    print(f"Lines removed: {lines_removed}")

    # Count total tokens and find the line with the most tokens in the cleaned file
    total_tokens = 0
    max_tokens = 0
    max_tokens_line = None

    with open(output_file, 'r') as outfile:
        for line in outfile:
            data = json.loads(line)
            conversations = data['conversations']
            line_tokens = sum(count_tokens(msg['value']) for msg in conversations)
            total_tokens += line_tokens
            if line_tokens > max_tokens:
                max_tokens = line_tokens
                max_tokens_line = line

    print(f"Total tokens in the cleaned file: {total_tokens}")
    print(f"Line with the most tokens: {max_tokens_line}")
    print(f"Number of tokens in the line with the most tokens: {max_tokens}")

if __name__ == '__main__':
    input_file = '/home/kquant/Documents/Datasets/SuperCleaned/SandevistanShard2.jsonl'
    output_file = '/home/kquant/Documents/Datasets/SuperCleaned/FINISH/SandevistanShard2.jsonl'
    deleted_file = '/home/kquant/Documents/Datasets/SuperCleaned/FINISH/SandevistanShard2_deleted.jsonl'
    process_dataset(input_file, output_file, deleted_file)
