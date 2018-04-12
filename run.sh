LOG_DIR="/home/trevor/coding/rnn_research/spectral-lstm/results";
BS=64;
HX=32;
EPC=25;
LAYERS=1;
DEFAULT_ARGS="--epochs ${EPC} --batch-size ${BS} --layers ${LAYERS}";

declare -a tasks=("add" "mul" "xor")

for i in 1 2 3 4 5 6 7 8 9 10
do
    for t in "${tasks[@]}"
    do

        EXP_PATH="${LOG_DIR}/gru_epoch${EPC}_bs${BS}_hx${HX}_task${t}/seed${i}/";
        echo $EXP_PATH;
        mkdir -p $EXP_PATH
        python3 main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --hx ${HX} --task ${t} --model-type gru &

        EXP_PATH="${LOG_DIR}/peephole_epoch${EPC}_bs${BS}_hx${HX}_task${t}/seed${i}/";
        echo $EXP_PATH;
        mkdir -p $EXP_PATH
        python3 main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --hx ${HX} --task ${t} --model-type phole &

        EXP_PATH="${LOG_DIR}/rnn_epoch${EPC}_bs${BS}_hx${HX}_task${t}/seed${i}/";
        echo $EXP_PATH;
        mkdir -p $EXP_PATH
        python3 main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --hx ${HX} --task ${t} --model-type rnn &

        EXP_PATH="${LOG_DIR}/lstm_epoch${EPC}_bs${BS}_hx${HX}_task${t}/seed${i}/";
        echo $EXP_PATH;
        mkdir -p $EXP_PATH
        python3 main.py ${DEFAULT_ARGS} --seed ${i} --log-dir ${EXP_PATH} --hx ${HX} --task ${t} --model-type lstm &
        wait;

    done
done
