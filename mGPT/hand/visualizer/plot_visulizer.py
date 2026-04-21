##### plot helper functions
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mGPT.hand.body_models.mano_xx import *


class Plot_visulizer():

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset


    def visualize_results_as_plots(self, model_output, model_input, batch, out_dir, curr_name, x_t=None):

        if model_output.shape[-1] == 6:
            return self.visualize_traj_results_as_plot( model_output, model_input, batch, out_dir, curr_name)

        # create the output
        gt_mano_full_pose = model_input[:1] # first sample
        gt_mano_params = mano_full_pose_to_mano_params(gt_mano_full_pose)

        pred_mano_full_pose = model_output[:1] # extract the first seq
        pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose) # outdict
        
        # curr_name = batch["seq_name"][0].split("/")[-1].split(".")[0] + "_ns%04d"%sample_id +"_step%04d"%step
        if batch.get("sentence", None) is not None:
            text = batch["sentence"][0]
        else:
            text = curr_name
        
        num_row = 3
        num_cols = 2
        fig, ax = plt.subplots(num_row, num_cols, figsize=(20, 16))
        set1_palette = iter(sns.color_palette("Set1", n_colors=10))
        fig.suptitle(f'{text}:Left and Right Hands)', fontsize=8)
        
        ### transl
        gt_hand_transl = torch.concatenate([gt_mano_params['lhand_transl'],  gt_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()
        pred_hand_transl = torch.concatenate([pred_mano_params['lhand_transl'],  pred_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()

        ### hand_pose
        gt_hand_global_orientaion = torch.concatenate([gt_mano_params['lhand_global_orientaion'],  gt_mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()
        pred_hand_global_orientaion = torch.concatenate([pred_mano_params['lhand_global_orientaion'],  pred_mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()

        ### hand_pose
        gt_hand_pose = torch.concatenate([gt_mano_params['lhand_pose'],  gt_mano_params['rhand_pose'] ], dim=0).detach().cpu().numpy()
        pred_hand_pose = torch.concatenate([pred_mano_params['lhand_pose'],  pred_mano_params['rhand_pose'] ], dim=0).detach().cpu().numpy()

        self.plot_mano_params_stats(ax, gt_hand_transl, gt_hand_global_orientaion,gt_hand_pose, next(set1_palette), name="GT" )
        self.plot_mano_params_stats(ax, pred_hand_transl, pred_hand_global_orientaion, pred_hand_pose, next(set1_palette), name="Pred" )

        out_file = os.path.join(out_dir, curr_name + ".png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {out_file}")

        return out_file
    
    def visualize_traj_results_as_plot(self, model_output, model_input, batch, out_dir, curr_name, x_t=None):

        # create the output
        gt_mano_full_pose = model_input[:1] # first sample
        pred_mano_full_pose = model_output[:1] # extract the first seq
        
        if pred_mano_full_pose.shape[-1] == 6: # only translation

            gt_mano_params = {}
            gt_mano_params["lhand_transl"] = model_input[:, :, :3]
            gt_mano_params["rhand_transl"] = model_input[:, :, 3:]
            
            pred_mano_params = {}
            pred_mano_params["lhand_transl"] = model_output[:, :, :3]
            pred_mano_params["rhand_transl"] = model_output[:, :, 3:]

            if x_t is not None:
                x_t_mano_params = {}
                x_t_mano_params["lhand_transl"] = x_t[:, :, :3]
                x_t_mano_params["rhand_transl"] = x_t[:, :, 3:]
        
        # elif pred_mano_full_pose.shape[-1] == 198: # trajector

        #     gt_mano_params = {}
        #     gt_mano_params["lhand_transl"] = model_input[:, :, :3]
        #     gt_mano_params["rhand_transl"] = model_input[:, :, 3:]

            
        #     pred_mano_params = {}
        #     pred_mano_params["lhand_transl"] = model_output[:, :, :3]
        #     pred_mano_params["rhand_transl"] = model_output[:, :, 3:]

        #     if x_t is not None:
        #         x_t_mano_params = {}
        #         x_t_mano_params["lhand_transl"] = x_t[:, :, :3]
        #         x_t_mano_params["rhand_transl"] = x_t[:, :, 3:]
            
        elif pred_mano_full_pose.shape[-1] == 198:
            gt_mano_params = mano_full_pose_to_mano_params(gt_mano_full_pose)
            pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose) # outdict

            if x_t is not None:
                x_t_mano_params = mano_full_pose_to_mano_params(x_t) # outdict

        if batch.get("sentence", None) is not None:
            text = batch["sentence"][0]
        else:
            text = curr_name
        
        num_row = 2
        num_cols = 1
        fig, ax = plt.subplots(num_row, num_cols, figsize=(10, 8))
        set1_palette = iter(sns.color_palette("Set1", n_colors=10))
        # fig.suptitle(f'{text}:Left and Right Hands)', fontsize=8)
        
        ### transl
        gt_hand_transl = torch.concatenate([gt_mano_params['lhand_transl'],  gt_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()
        pred_hand_transl = torch.concatenate([pred_mano_params['lhand_transl'],  pred_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()

 
        # ### hand_pose
        # gt_hand_global_orientaion = torch.concatenate([gt_mano_params['lhand_global_orientaion'],  gt_mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()
        # pred_hand_global_orientaion = torch.concatenate([pred_mano_params['lhand_global_orientaion'],  pred_mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()

        # ### hand_pose
        # gt_hand_pose = torch.concatenate([gt_mano_params['lhand_pose'],  gt_mano_params['rhand_pose'] ], dim=0).detach().cpu().numpy()
        # pred_hand_pose = torch.concatenate([pred_mano_params['lhand_pose'],  pred_mano_params['rhand_pose'] ], dim=0).detach().cpu().numpy()

        self.plot_mano_params_stat_dict(ax, {"transl":gt_hand_transl}, next(set1_palette), name="GT" )
        self.plot_mano_params_stat_dict(ax, {"transl":pred_hand_transl}, next(set1_palette), name="Pred" )


        if x_t is not None:
            x_t_hand_transl = torch.concatenate([x_t_mano_params['lhand_transl'],  x_t_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()
            self.plot_mano_params_stat_dict(ax, {"transl":x_t_hand_transl}, next(set1_palette), name="X_t" )




        out_file = os.path.join(out_dir, curr_name + ".png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {out_file}")

        return out_file

    def visualize_traj_results_as_plot_with_orientation(self, model_output, model_input, batch, out_dir, curr_name, x_t=None):

        # create the output
        gt_mano_full_pose = model_input[:1] # first sample
        pred_mano_full_pose = model_output[:1] # extract the first seq
        
        if pred_mano_full_pose.shape[-1] == 6: # only translation

            gt_mano_params = {}
            gt_mano_params["lhand_transl"] = model_input[:1, :, :3]
            gt_mano_params["rhand_transl"] = model_input[:1, :, 3:]
            
            pred_mano_params = {}
            pred_mano_params["lhand_transl"] = model_output[:1, :, :3]
            pred_mano_params["rhand_transl"] = model_output[:1, :, 3:]

            if x_t is not None:
                x_t_mano_params = {}
                x_t_mano_params["lhand_transl"] = x_t[:, :, :3]
                x_t_mano_params["rhand_transl"] = x_t[:, :, 3:]
        
        if pred_mano_full_pose.shape[-1] == 18: # only translation

            gt_mano_params = {}

            gt_mano_params["rhand_global_orientaion"] = model_input[:1, :, :6]
            gt_mano_params["lhand_transl"] = model_input[:1, :, 6:9]
            gt_mano_params["lhand_global_orientaion"] = model_input[:1, :, 9:15]
            gt_mano_params["rhand_transl"] = model_input[:1, :, 15:18]
            
            pred_mano_params = {}
            pred_mano_params["rhand_global_orientaion"] = model_output[:1, :, :6]
            pred_mano_params["lhand_transl"] = model_output[:1, :, 6:9]
            pred_mano_params["lhand_global_orientaion"] = model_output[:1, :, 9:15]
            pred_mano_params["rhand_transl"] = model_output[:1, :, 15:18]
            

            # if x_t is not None:
            #     x_t_mano_params = {}
            #     x_t_mano_params["lhand_transl"] = x_t[:, :, :3]
            #     x_t_mano_params["rhand_transl"] = x_t[:, :, 3:]
            
        elif pred_mano_full_pose.shape[-1] == 198:
            gt_mano_params = mano_full_pose_to_mano_params(gt_mano_full_pose)
            pred_mano_params = mano_full_pose_to_mano_params(pred_mano_full_pose) # outdict

            if x_t is not None:
                x_t_mano_params = mano_full_pose_to_mano_params(x_t) # outdict

        if batch.get("sentence", None) is not None:
            text = batch["sentence"][0]
        else:
            text = curr_name
        
        num_row = 2
        num_cols = 2
        fig, ax = plt.subplots(num_row, num_cols, figsize=(15, 8))
        set1_palette = iter(sns.color_palette("Set1", n_colors=10))

        ### transl
        gt_hand_transl = torch.concatenate([gt_mano_params['lhand_transl'],  gt_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()
        pred_hand_transl = torch.concatenate([pred_mano_params['lhand_transl'],  pred_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()

         ### hand_pose
        gt_hand_global_orientaion = torch.concatenate([gt_mano_params['lhand_global_orientaion'],  gt_mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()
        pred_hand_global_orientaion = torch.concatenate([pred_mano_params['lhand_global_orientaion'],  pred_mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()

        self.plot_mano_params_stat_dict(ax, {"transl":gt_hand_transl, "Orient": gt_hand_global_orientaion}, next(set1_palette), name="GT" )
        self.plot_mano_params_stat_dict(ax, {"transl":pred_hand_transl, "Orient": pred_hand_global_orientaion }, next(set1_palette), name="Pred" )

        if x_t is not None:
            x_t_hand_transl = torch.concatenate([x_t_mano_params['lhand_transl'],  x_t_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()
            self.plot_mano_params_stat_dict(ax, {"transl":x_t_hand_transl}, next(set1_palette), name="X_t" )

        # Add text at the bottom of the figure
        fig.text(0.5, 0.02, text,  # 0.02 is the distance from bottom
                horizontalalignment='center',
                verticalalignment='center',
                wrap=True,
                fontsize=14)


        out_file = os.path.join(out_dir, curr_name + ".png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {out_file}")

        return out_file


    def plot_mano_params_stat_dict(self, ax, params, color, name, plot_std=False):
        """
        Plot one set of MANO parameters statistics on given axes.
        
        Args:
            ax: Matplotlib axes object (4x2 subplot axes)
            trans: Translation parameters (Bs x T x 3)
            rot: Rotation parameters (Bs x T x 3)
            hand_pose: Hand pose parameters (Bs x T x 45)
            betas: Shape parameters (Bs x T x 10)
            color: Color for the plots
            name: Label name for the legend
        """
        hand_labels = ['Left Hand', 'Right Hand']
        param_names =  ['Translation', 'Rotation', 'Hand Pose']

        # Get time steps
        ip_ = list(params.values())[0]
        time_steps = np.arange(ip_.shape[1])

        if len(params) > 1:
            for hand_idx in range(2):  # Loop over hands
                for param_idx, (param_name, param_data) in enumerate(params.items()):
                    
                    # Calculate statistics
                    mean_param = np.mean(param_data[hand_idx], axis=1)
                    std_param = np.std(param_data[hand_idx], axis=1)
                    
                    # Plot parameters
                    ax[param_idx, hand_idx].plot(time_steps, mean_param, 
                                            color=color, 
                                            label=name, 
                                            alpha=0.7)
                    
                    if plot_std:
                        ax[param_idx, hand_idx].fill_between(time_steps,
                                                        mean_param - std_param,
                                                        mean_param + std_param,
                                                        color=color, 
                                                        alpha=0.1)
                    
                    # Set titles and labels
                    ax[param_idx, hand_idx].set_title(f'{param_names[param_idx]} Parameters - {hand_labels[hand_idx]}')
                    ax[param_idx, hand_idx].set_ylabel('Value')
                    if param_idx == len(params) - 1:
                        ax[param_idx, hand_idx].set_xlabel('Time Steps')
                    ax[param_idx, hand_idx].legend()
        else:        
            for hand_idx in range(2):  # Loop over hands
                for param_idx, (param_name, param_data) in enumerate(params.items()):
                    # Calculate statistics
                    mean_param = np.mean(param_data[hand_idx], axis=1)
                    std_param = np.std(param_data[hand_idx], axis=1)
                    
                    # Plot parameters
                    # ax[hand_idx, param_idx].plot(time_steps, mean_param, 
                    ax[hand_idx].plot(time_steps, mean_param, 
                                            color=color, 
                                            label=name, 
                                            alpha=0.7)
                    
                    if plot_std:
                        ax[hand_idx].fill_between(time_steps,
                                                        mean_param - std_param,
                                                        mean_param + std_param,
                                                        color=color, 
                                                        alpha=0.1)
                    
                    # Set titles and labels
                    ax[hand_idx].set_title(f'{param_names[param_idx]} Parameters - {hand_labels[hand_idx]}')
                    ax[hand_idx].set_ylabel('Value')
                    if param_idx == len(params) - 1:
                        ax[hand_idx].set_xlabel('Time Steps')
                    ax[hand_idx].legend()
            

    def plot_mano_params_stats(self, ax, trans, rotation, hand_pose, color, name, plot_std=False):
        """
        Plot one set of MANO parameters statistics on given axes.
        
        Args:
            ax: Matplotlib axes object (4x2 subplot axes)
            trans: Translation parameters (Bs x T x 3)
            rot: Rotation parameters (Bs x T x 3)
            hand_pose: Hand pose parameters (Bs x T x 45)
            betas: Shape parameters (Bs x T x 10)
            color: Color for the plots
            name: Label name for the legend
        """
        hand_labels = ['Left Hand', 'Right Hand']
        param_names = ['Global_transl', 'Global_rot', 'Hand Pose']
        
        # Get time steps
        time_steps = np.arange(trans.shape[1])
        
        # Parameters to plot
        params = [
            (trans, 'Global_transl'),
            (rotation, 'Global_rot'),
            (hand_pose, 'hand_pose'),
        ]

        for hand_idx in range(2):  # Loop over hands
            for param_idx, (param_data, param_name) in enumerate(params):
                
                # Calculate statistics
                mean_param = np.mean(param_data[hand_idx], axis=1)
                std_param = np.std(param_data[hand_idx], axis=1)
                
                # Plot parameters
                ax[param_idx, hand_idx].plot(time_steps, mean_param, 
                                        color=color, 
                                        label=name, 
                                        alpha=0.7)
                
                if plot_std:
                    ax[param_idx, hand_idx].fill_between(time_steps,
                                                    mean_param - std_param,
                                                    mean_param + std_param,
                                                    color=color, 
                                                    alpha=0.1)
                
                # Set titles and labels
                ax[param_idx, hand_idx].set_title(f'{param_names[param_idx]} Parameters - {hand_labels[hand_idx]}')
                ax[param_idx, hand_idx].set_ylabel('Value')
                if param_idx == len(params) - 1:
                    ax[param_idx, hand_idx].set_xlabel('Time Steps')
                ax[param_idx, hand_idx].legend()


    def create_plot_with_colour(self, n_feats, n_colors=10):
        
        if n_feats == 6:
            num_row=2
            num_col=1
            figsize = (10, 8) 
        elif n_feats == 18:
            num_row = 2
            num_col=2
            figsize = (15, 8)
        else:
            num_row = 3
            num_col=2
            figsize = (20, 16) 

        fig, ax = plt.subplots(nrows=num_row, ncols=num_col, figsize=figsize)
        set1_palette = iter(sns.color_palette("Set1", n_colors=n_colors))
        return fig, ax, set1_palette
    
    def iterative_plot_base(self, ax, params, batch, color, name):

        if params.shape[-1] == 6:
            return self.iterative_traj_results_as_plot(ax, params, batch, color, name)
        elif params.shape[-1] == 18: # only translation
            return self.iterative_visulize_model_params_as_plot_with_orientation(ax, params, batch, color, name)
        else:
            self.iterative_visulize_model_params_as_plot(ax, params, batch, color, name)

    def iterative_visulize_model_params_as_plot(self, ax, params, batch, color, name):

        # create the output
        mano_full_pose = params[:1] # first sample
        mano_params = mano_full_pose_to_mano_params(mano_full_pose)
        
        ### transl
        hand_transl = torch.concatenate([mano_params['lhand_transl'],  mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()

        ### hand_pose
        hand_global_orientaion = torch.concatenate([mano_params['lhand_global_orientaion'],  mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()

        ### hand_pose
        hand_pose = torch.concatenate([mano_params['lhand_pose'],  mano_params['rhand_pose'] ], dim=0).detach().cpu().numpy()

        self.plot_mano_params_stat_dict(ax, {"transl":hand_transl, "rotation":hand_global_orientaion,"handpose":hand_pose  }, color, name=name)


    def iterative_visulize_model_params_as_plot_with_orientation(self, ax, params, batch, color, name):

        # create the output
        mano_params = {}
        mano_params["rhand_global_orientaion"] = params[:1, :, :6]
        mano_params["rhand_transl"] = params[:1, :, 6:9]

        mano_params["lhand_global_orientaion"] = params[:1, :, 9:15]
        mano_params["lhand_transl"] = params[:1, :, 15:18]

        ### transl
        hand_transl = torch.concatenate([mano_params['lhand_transl'],  mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()

        ### hand_pose
        hand_global_orientaion = torch.concatenate([mano_params['lhand_global_orientaion'],  mano_params['rhand_global_orientaion'] ], dim=0).detach().cpu().numpy()
            

        # hand_pose = torch.concatenate([mano_params['lhand_pose'],  mano_params['rhand_pose'] ], dim=0).detach().cpu().numpy()

        self.plot_mano_params_stat_dict(ax, {"transl":hand_transl, "rotation":hand_global_orientaion  }, color, name=name)


    def iterative_traj_results_as_plot( self, ax, params, batch, color, name):

        transl_mano_params = {}
        transl_mano_params["lhand_transl"] = params[:, :, :3]
        transl_mano_params["rhand_transl"] = params[:, :, 3:]

        transl_mano_params = torch.concatenate([transl_mano_params['lhand_transl'],  transl_mano_params['rhand_transl'] ], dim=0).detach().cpu().numpy()
        self.plot_mano_params_stat_dict(ax, {"transl":transl_mano_params}, color, name=name)

    
    def write_plot_as_png(self, fig, out_dir, name, text=None):

        if text is not None:
            # Add text at the bottom of the figure
            fig.text(0.5, 0.02, text,  # 0.02 is the distance from bottom
                    horizontalalignment='center',
                    verticalalignment='center',
                    wrap=True,
                    fontsize=14)

        out_file = os.path.join(out_dir, name + ".png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {out_file}")

        return out_file
