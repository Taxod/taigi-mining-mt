from datasets import load_dataset
import hanzidentifier

# Load the dataset with zho_Hani configuration
ds = load_dataset('cis-lmu/Glot500', 'zho_Hani', split='train', streaming=True)

target_count = 640000
count = 0

print('Starting to extract Traditional Chinese sentences using hanzidentifier...')

with open('zho_hant_hanz_640k.txt', 'w', encoding='utf-8') as f:
    for entry in ds:
        text = entry['text']

        # Check if text contains Chinese characters AND is strictly Traditional Chinese
        if hanzidentifier.has_chinese(text) and hanzidentifier.is_traditional(text):
            f.write(text.replace('\n', ' ') + '\n')
            count += 1

            if count % 10000 == 0:
                print(f'Collected {count} sentences...')

            if count >= target_count:
                break

if count < target_count:
    print(f'Search finished. Only found {count} sentences.')
else:
    print('Download complete! File saved as zho_hant_hanz_640k.txt')