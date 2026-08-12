import os
import re

directory = r"C:\AI\AI-Engineering-Docs\Fine-Tuning"

for filename in os.listdir(directory):
    if not filename.endswith(".md"):
        continue
        
    filepath = os.path.join(directory, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    in_math_block = False
    
    for i, line in enumerate(lines):
        if line.strip() == "$$":
            if not in_math_block:
                # Starting math block: ensure empty line before
                if len(new_lines) > 0 and new_lines[-1].strip() != "":
                    new_lines.append("")
                new_lines.append(line)
                in_math_block = True
            else:
                # Ending math block: ensure empty line after
                new_lines.append(line)
                if i + 1 < len(lines) and lines[i+1].strip() != "":
                    new_lines.append("")
                in_math_block = False
        else:
            if in_math_block:
                # Replace exactly 2 backslashes with 4 backslashes for markdown escaping
                line = re.sub(r'(?<!\\)\\\\(?!\\)', r'\\\\\\\\', line)
            new_lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write('\n'.join(new_lines))

print("Formatting applied successfully.")
