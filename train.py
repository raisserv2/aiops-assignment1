import argparse
import subprocess
import mlflow
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hidden_layers = tuple(int(x) for x in args.hidden.split(","))

    # Load MNIST from DVC-tracked local file
    print("Loading MNIST from mnist_data.csv...")
    df = pd.read_csv("mnist_data.csv")
    X = df.drop(columns=["target"]).values / 255.0
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mnist-mlp")

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    with mlflow.start_run(run_name=f"lr{args.lr}_h{args.hidden}_bs{args.batch_size}"):
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("hidden_layers", args.hidden)
        mlflow.log_param("max_epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("seed", args.seed)
        mlflow.set_tag("git_commit", git_commit)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=args.lr,
            max_iter=args.epochs,
            batch_size=args.batch_size,
            random_state=args.seed,
        )

        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("final_accuracy", acc)
        mlflow.sklearn.log_model(
            model, "model",
            serialization_format="pickle",
            registered_model_name="my-classifier",
        )
        print(f"Done. Final accuracy: {acc:.4f}")
        print(f"Git commit: {git_commit}")

if __name__ == "__main__":
    main()
