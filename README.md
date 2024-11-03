# bondbidhie2024_algorithm Algorithm

The source code for the algorithm container for
bondbidhie2024_algorithm_segnet

Prepared by Rina Bao, 10/16/2024

The source code for the algorithm container for bondbidhie2024_algorithm_outcomenet.

Before running, you will need a local docker installation. For more details, please read grand-challenge documents https://grand-challenge.org/documentation/automated-evaluation/ and https://comic.github.io/evalutils/usage.html

process.py is the main function for generating prediction results.

Noted that when your algorithm is run on grand challenge <local_path>/case1/ will be mapped to /input/. Then a separate run will be made with <local_path>/case2/ mapped to /input/. This allows grand challenge to execute all the jobs in parallel on their cloud infrastructure. For simplicity, you can include one case in your test data when you test locally. The platform will handle multiple cases. Predict should only handle one case.

Please follow these steps to run it on the local machine.

(1)./build.sh

(2)./test.sh

In order to generate the results locally and test your codes, In test.sh, use this piece of codes:

docker run --rm
--memory="${MEM_LIMIT}"
--memory-swap="${MEM_LIMIT}"
--network="none"
--cap-drop="ALL"
--security-opt="no-new-privileges"
--shm-size="128m"
--pids-limit="256"
-v $SCRIPTPATH/test/:/input/
-v $SCRIPTPATH/output/:/output/
bondbidhie2024_algorithm_segnet

But for uploading algorithm docker to the grand challenge server, please use the codes that I provided in test.sh.

docker run --rm
--memory="${MEM_LIMIT}"
--memory-swap="${MEM_LIMIT}"
--network="none"
--cap-drop="ALL"
--security-opt="no-new-privileges"
--shm-size="128m"
--pids-limit="256"
-v $SCRIPTPATH/test/:/input/
-v bondbidhie2024_algorithm_segnet-output-$VOLUME_SUFFIX:/output/
bondbidhie2024_algorithm_segnet

(3) ./export.sh Running ./export.sh, and submitting the generated zip file of the algorithm docker.
