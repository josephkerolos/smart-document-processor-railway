#!/usr/bin/env python3
"""Fix the null issue in main_enhanced_v9.py"""

import re

# Read the file
with open('main_enhanced_v9.py', 'r') as f:
    content = f.read()

# Find all instances of standalone 'null' (not in strings)
# This regex looks for 'null' that's not part of a larger word and not in quotes
pattern = r'\breturn\s+null\b'
matches = re.findall(pattern, content)
print(f"Found {len(matches)} instances of 'return null'")

# Replace them
content = re.sub(r'\breturn\s+null\b', 'return None', content)

# Also check for 'false' instead of 'False'
pattern2 = r'\breturn\s+false\b'
matches2 = re.findall(pattern2, content)
print(f"Found {len(matches2)} instances of 'return false'")
content = re.sub(r'\breturn\s+false\b', 'return False', content)

# Check for any remaining 'null' not in strings
lines = content.split('\n')
for i, line in enumerate(lines):
    # Skip comments and strings
    if not line.strip().startswith('#') and not line.strip().startswith('"') and not line.strip().startswith("'"):
        if 'null' in line and 'null' not in repr(line):
            print(f"Line {i+1}: {line.strip()}")

# Write the fixed version
with open('main_enhanced_v9_fixed.py', 'w') as f:
    f.write(content)

print("Created main_enhanced_v9_fixed.py")