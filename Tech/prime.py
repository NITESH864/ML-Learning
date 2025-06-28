n=int(input("Enter the number:"))

if n>1:
	for i in range(2,10):
		if n%i==0:
			print("Prime number:",n)
		else:
			print("Not Prime",n)
	
	