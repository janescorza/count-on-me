
import random

def main():
    while True:
        selected_number = random.randint(0, 5)
        user_input = input(f"Intput the number {selected_number} which is between 0 and 5 (press 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Thanks for counting on me ;)")
            break
        try:
            user_guess = int(user_input)
            if user_guess == selected_number:
                print("Congratulations! You guessed correctly.")
            else:
                print(f"Wrong guess. The correct number was {selected_number}.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

if __name__ == "__main__":
    main()
