# Test 1:
"""
with open("requirements.txt", "r",) as f:
    required_packages = []
    for i in f.readlines():
        if i[:-1] != '' and i[:-1]!='-e.':
            required_packages.append(i[:-1])

print(required_packages)
"""

