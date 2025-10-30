import dataset, modules, train
from config import Config

if __name__ == "__main__":
    config = Config.getInstance()

    print("Preparing dataset...")
    data = dataset.Dataset(
        config['data_dir'],
        config['class_split'],
        config['train_split']
    )

    print("Constructing model...")
    model = modules.construct_classifier()
    print("Compiling model...")
    modules.compile_model(model, config['learning_rate'])

    # print("Loading weights if available...")
    # train.load_weights(model)

    train.train(
        model, 
        data,
        config['epochs']
    )
    
    train.validate(
        model,
        data
    )
