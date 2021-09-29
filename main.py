import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='The selected model', choices=[])

if __name__ == '__main__':
    args = parser.parse_args()
    print("It works!")
