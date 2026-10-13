namespace YoloSharp.Utils
{
    internal class EarlyStopping
    {
        float best_fitness;         
        int best_epoch;
        float patience;              
        bool possible_stop;          

        internal EarlyStopping(int patience = 50)
        {
            this.best_fitness = float.NegativeInfinity;   
            this.best_epoch = 0;
            this.patience = patience;
            this.possible_stop = false;
        }

        internal bool ShouldStop(float fitness, int epoch)
        {
            if (fitness > this.best_fitness)
            {
                this.best_epoch = epoch;
                this.best_fitness = fitness;
            }
            int delta = epoch - this.best_epoch;          // epochs without improvement
            this.possible_stop = delta >= (this.patience - 1);
            bool stop = delta >= this.patience;
            if (stop)
            {
                string str = string.Format(
                                $"Training stopped early as no improvement observed in last {this.patience} epochs. " +
                                $"Best results observed at epoch {this.best_epoch}, best model saved as best.bin.\n" +
                                $"To update EarlyStopping(patience={this.patience}) pass a new patience value, " +
                                $"i.e. `patience=300` or use `patience=0` to disable EarlyStopping.");
                Console.WriteLine(str);
            }
            return stop;
        }
    }
}