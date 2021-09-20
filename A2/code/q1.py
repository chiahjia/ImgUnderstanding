from matplotlib import pyplot as plt
from matplotlib import collections as matcoll


x_list = [-2, -1, 0, 1, 2]

fx_list = [4, 1, 5, 1, 4]

x_ans = []
y_ans = []

for index in range(len(x_list)):
	if (index + 1) == len(x_list):
		x_ans.append(x_list[index])
		y_ans.append(fx_list[index])
		break

	x1 = x_list[index]
	x2 = x_list[index+1]

	fx1 = fx_list[index]
	fx2 = fx_list[index+1]

	factor = 0.25
	counter = 1

	x_ans.append(x1)
	y_ans.append(fx1)

	for counter in range(4):
		x = x1 + factor * counter
		fx = (x - x2)/(x1 - x2) * fx1 + (x1 - x)/(x1 - x2) * fx2

		x_ans.append(x)
		y_ans.append(fx)

		counter += 1

	counter = 1
	x_ans.append(x2)
	y_ans.append(fx2)


#x = np.arange(1,13)
#y = [15,14,15,18,21,25,27,26,24,20,18,16]

lines = []
for i in range(len(x_ans)):
    pair=[(x_ans[i],0), (x_ans[i], y_ans[i])]
    lines.append(pair)

linecoll = matcoll.LineCollection(lines)
fig, ax = plt.subplots()
ax.add_collection(linecoll)

print(x_ans)
print()
print(y_ans)

plt.scatter(x_ans,y_ans)

plt.xticks(x_ans)
plt.ylim(0,6)

plt.show()





