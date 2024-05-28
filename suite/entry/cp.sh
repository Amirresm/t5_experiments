#!/usr/bin/env python3

import os

#go through each directory 

for root, dirs, files in os.walk("."):
	for file in files:
		if "t5l" in file or "cp.sh" in file:
			continue
		if file.endswith(".sh"):
			file_path = os.path.join(root, file)
			new_file = file_path.replace("t5", "codet5")

			print("Moving {} to {}".format(file_path, new_file))
			#run the cp command
			os.system("cp {} {}".format(file_path, new_file))

