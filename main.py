from projectIM2022_q1 import apply_ex1_detection
from projectIM2022_q2 import apply_ex2_detection

print("Insert q or Q to quit mate!")
while True:
    ex_number = input("Please select which exercise you wish to activate (insert 1 or 2 accordingly)\n>>> ")

    if ex_number == str(1):
        apply_ex1_detection()
    elif ex_number == str(2):
        apply_ex2_detection()
    elif ex_number == "q" or ex_number == "Q":
        print("Goodbye, we hope you had fun using our program \U0001f604")
        break
