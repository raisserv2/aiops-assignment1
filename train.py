import argparse
import mlflow
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hidden_layers = tuple(int(x) for x in args.hidden.split(","))

    # Load MNIST
    print("Loading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    X = X / 255.0  # normalize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mnist-mlp")

    with mlflow.start_run(run_name=f"lr{args.lr}_h{args.hidden}_bs{args.batch_size}"):
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("hidden_layers", args.hidden)
        mlflow.log_param("max_epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("seed", args.seed)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=args.lr,
            max_iter=1,          # we loop manually to log per-epoch
            batch_size=args.batch_size,
            random_state=args.seed,
            warm_start=True,     # lets us loop epochs manually
        )

        for epoch in range(1, args.epochs + 1):
            model.fit(X_train, y_train)
            train_loss = model.loss_
            val_acc = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"Epoch {epoch}: loss={train_loss:.4f} val_acc={val_acc:.4f}")

        final_acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("final_accuracy", final_acc)
        mlflow.sklearn.log_model(model, "model", serialization_format="pickle")
        print(f"Done. Final accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    main()
