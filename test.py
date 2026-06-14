locations = []

with open("test.txt", "r") as file:
    content = file.readlines()
    for line in content:
        if line.strip() not in locations:
            locations.append(line.strip())

print(locations)