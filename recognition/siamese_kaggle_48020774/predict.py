import dataset, modules, train
import json

if __name__ == "__main__":
    config = json.load(open('recognition/siamese_kaggle_48020774/utility/config.json'))

    data = dataset.Dataset(
        config['data_dir'],
        config["class_split"],
        config['train_size'],
        config['test_size'],
        config['validate_size']
    )

    model = modules.construct_classifier()
    modules.compile_model(model, config['learning_rate'])

    train.train(
        model,
        data.GenerateTrainSet(200),
        data.GenerateTestSet(100)
    )