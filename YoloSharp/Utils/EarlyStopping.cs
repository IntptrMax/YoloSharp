namespace YoloSharp.Utils
{
    internal class EarlyStopping
    {
        float best_fitness = 0.0f;  // i.e. mAP
        int best_epoch = 0;
        float patience = float.PositiveInfinity; // epochs to wait after fitness stops improving to stop
        bool possible_stop = false;  // possible stop may occur next epoch

        internal EarlyStopping(int patience = 50)
        {
            this.best_fitness = 0.0f; // i.e. mAP
            this.best_epoch = 0;
            this.patience = patience;  // epochs to wait after fitness stops improving to stop
            this.possible_stop = false;  // possible stop may occur next epoch
        }

        internal bool ShouldStop(float fitness, int epoch)
        {
            if (fitness > this.best_fitness || this.best_fitness == 0)
            {
                this.best_epoch = epoch;
                this.best_fitness = fitness;
            }
            int delta = epoch - this.best_epoch;  // epochs without improvement
            this.possible_stop = delta >= (this.patience - 1);  // possible stop may occur next epoch
            bool stop = delta >= this.patience;  // stop training if patience exceeded
            if (stop)
            {
                string str = string.Format(
                                $"Training stopped early as no improvement observed in last {this.patience} epochs. " +
                                $"Best results observed at epoch {this.best_epoch}, best model saved as best.pt.\n" +
                                $"To update EarlyStopping(patience={this.patience}) pass a new patience value, " +
                                $"i.e. `patience=300` or use `patience=0` to disable EarlyStopping.");
                Console.WriteLine(str);
            }
            return stop;
        }
    }
}
