using TorchSharp;
using TorchSharp.Modules;

namespace YoloSharp.Modules
{
    internal class Activation
    {
        /// <summary>
        /// Unified activation function module from AGLU.
        /// </summary>
        /// <remarks>
        /// This class implements a parameterized activation function with learnable parameters lambda and kappa, based on the AGLU(Adaptive Gated Linear Unit) approach.
        /// </remarks>
        /// <see cref="https://github.com/kostas1515/AGLU"/>
        internal class AGLU : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Softplus act;
            private readonly Parameter lambd;
            private readonly Parameter kappa;

            /// <summary>
            /// Initialize the Unified activation function with learnable parameters.`
            /// </summary>
            /// <param name="device"></param>
            /// <param name="dtype"></param>
            internal AGLU(torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(AGLU))
            {
                this.act = torch.nn.Softplus(beta: -1.0);
                this.lambd = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(1, device: device, dtype: dtype)));  // lambda parameter
                this.kappa = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(1, device: device, dtype: dtype))); // kappa parameter
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                torch.Tensor lam = torch.clamp(this.lambd, min: 0.0001);  // Clamp lambda to avoid division by zero
                return torch.exp((1 / lam) * this.act.forward((this.kappa * x) - torch.log(lam)));
            }
        }
    }
}
