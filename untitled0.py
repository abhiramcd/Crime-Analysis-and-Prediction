
print("1.Add")
print("2.Subtract")
print("3.Multiply")
print("4.Divide")

select = input("Enter choice(1/2/3/4): ")

if select in ('1', '2', '3', '4'):
    x = float(input("Enter first number: "))
    y = float(input("Enter second number: "))
    myfile=open("file.txt","w")

    if select == '1':
        print(x, "+", y, "=", myfile.write(x+y))

    elif select == '2':
        print(x, "-", y, "=", myfile.write(x-y))

    elif select == '3':
        print(x, "*", y, "=", myfile.write(x*y))

    elif select == '4':
        print(x, "/", y, "=", myfile.write(x/y))