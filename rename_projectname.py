import argparse


parser = argparse.ArgumentParser(description='Rename project name in the main.py file')
parser.add_argument('project_name', help='Project name')

args = parser.parse_args()

with open('src/main.py', 'r') as f:
    content = f.read()

content = content.replace('project_name', args.project_name)

with open('src/main.py', 'w') as f:
    f.write(content)
