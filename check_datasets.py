import json

# Analyze datasets/datasets/train.jsonl
tasks_count_1 = {}
with open('datasets/datasets/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        task = data.get('task', 'unknown')
        tasks_count_1[task] = tasks_count_1.get(task, 0) + 1

print('=== datasets/datasets/train.jsonl ===')
print(f'Total samples: {sum(tasks_count_1.values())}')
for task, count in sorted(tasks_count_1.items()):
    print(f'  {task}: {count}')

# Analyze downloaded_datasets/train.jsonl
tasks_count_2 = {}
with open('datasets/downloaded_datasets/train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        task = data.get('task', 'unknown')
        tasks_count_2[task] = tasks_count_2.get(task, 0) + 1

print('\n=== datasets/downloaded_datasets/train.jsonl ===')
print(f'Total samples: {sum(tasks_count_2.values())}')
for task, count in sorted(tasks_count_2.items()):
    print(f'  {task}: {count}')

# Check expected data from architecture
print('\n=== Expected Data (from architecture.md) ===')
expected = {
    'fluency': 1500,
    'vocabulary': 2500,
    'grammar': 9200,
    'dialogue': 3500,
}
print(f'Total expected: {sum(expected.values())} samples')
for task, count in sorted(expected.items()):
    print(f'  {task}: {count}')
