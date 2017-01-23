import pandas as pa
import matplotlib.pyplot as plt

numOfDays = 20
network = pa.read_csv("../Data/network_backup_dataset.csv", header=0)
dict = {"Monday": "1", "Tuesday": "2", "Wednesday": "3", "Thursday": "4", "Friday": "5", "Saturday": "6", "Sunday": "7"}

workflows = pa.unique(network["Work-Flow-ID"])
colors = {workflows[0]: "red", workflows[1]: "orange", workflows[2]: "yellow", workflows[3]: "green",
          workflows[4]: "blue"}

for i in dict:
    network["Day of Week"] = [s.replace(i, dict[i]) for s in network["Day of Week"]]
network["Day of Week"] = [int(s) for s in network["Day of Week"]]

# plot figure: Size of Backup over Days
plt.figure()
for w in workflows:
    networkByWorkflow = network[network["Work-Flow-ID"] == w]
    size = [0.0] * (numOfDays + 1)
    for row in networkByWorkflow.values:
        day = int(((row[0] - 1) * 7) + row[1])
        if (day <= numOfDays):
            size[day] += float(row[5])

    plt.plot(range(len(size)), size, colors[w])
    plt.title("Size of Backup over Days")
    plt.xlabel("Days")
    plt.ylabel("Size of Backup")

# plot figure: Backup Time over Days
plt.figure()
for w in workflows:
    networkByWorkflow = network[network["Work-Flow-ID"] == w]
    time = [0.0] * (numOfDays + 1)
    for row in networkByWorkflow.values:
        day = int(((row[0] - 1) * 7) + row[1])
        if (day <= numOfDays):
            time[day] += float(row[6])

    plt.plot(range(len(time)), time, colors[w])
    plt.title("Backup Time over Days")
    plt.xlabel("Days")
    plt.ylabel("Backup Time")
plt.show()
