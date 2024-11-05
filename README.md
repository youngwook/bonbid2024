# bondbidhie2024_algorithm Algorithm

The source code for the algorithm container for bondbidhie2024_algorithm_segnet.

Prepared Rongxu xu, 11/05/2024

Before running, you will need a local docker installation. Please refer to [docker](https://docs.docker.com/engine/install/).

process.py is the main function for generating prediction results.

Please follow these steps to run it on the local machine.

(1)./build.sh

(2)./test.sh

In order to generate the results locally and test your codes, In test.sh, use this piece of codes:
Where $SCRIPTPATH/test/ refer to local test folder and /input/ indicate container folder.

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

(3) ./export.sh generating zip file of the algorithm docker as bondbidhie2024_algorithm_segnet.tar.gz
