import dataset, modules, train
import json

if __name__ == "__main__":
    config = json.load(open('recognition/siamese_kaggle_48020774/utility/config.json'))

    data = dataset.Dataset(
        config['data_dir'],
        config["class_split"],
        config['train_split']
    )

    model = modules.construct_classifier()
    modules.compile_model(model, config['learning_rate'])

    train.load_weights(model)

    # train.validate(
    #     model,
    #     data.GenerateTestSet(100)
    # )

    train_set = data.GenerateTrainSet(200)
    test_set = data.GenerateTestSet(100)
    train.train(
        model,
        train_set,
        test_set
    )