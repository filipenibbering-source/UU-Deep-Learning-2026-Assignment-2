hidden_sizes=(8 32 64)
num_layers=(1 4)
bidirectionals=(true false)


# LSTM
for hidden_size in "${hidden_sizes[@]}"; do
    for bidirectional in "${bidirectionals[@]}"; do
        for num_layer in "${num_layers[@]}"; do
cat > "baseline_lstm_${hidden_size}_${bidirectional}_${num_layer}.yaml" <<EOL

model: lstm
is_baseline: true
hidden_size: ${hidden_size}
bidirectional: ${bidirectional}
num_layers: ${num_layer}
batch_size: 4
EOL

        done
    done
done

# resnet-18
cat > "baseline_resnet18.yaml" <<EOL
model: resnet18
is_baseline: true
batch_size: 4
EOL

# resnet-50
cat > "baseline_resnet50.yaml" <<EOL
model: resnet50
is_baseline: true
batch_size: 4
EOL

# eegnet
cat > "baseline_eegnet.yaml" <<EOL
model: eegnet
is_baseline: true
batch_size: 4
EOL