import torch
import numpy as np

from visualize import save_ratemaps
import os

class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=True):
        
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []
        self.val_loss = []


        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))
    
    def validation_step(self, test_inputs):
        ''' 
        Validate the model on a given test batch.

        Args:
            test_inputs: Test batch to compute the validation loss.
    
        Returns:
            val_loss: Validation loss for this test batch.
        '''
        with torch.no_grad():
            val_loss = self.model.compute_loss(test_inputs)
        return val_loss.item()

    def train_step(self, inputs):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
    
        Returns:
            loss: Avg. MI_loss for this training batch.
        '''
        
        self.model.zero_grad()
        loss = self.model.compute_loss(inputs)
        loss_value = loss.item()
        
        loss.backward()
        self.optimizer.step()

        return loss_value

    def train(self, n_epochs: int = 1000, n_steps=10, stopping_criterion = False, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()
        test_inputs, _ = self.trajectory_generator.get_test_batch()
        
        # tbar = tqdm(range(n_steps), leave=False)
        
        best_loss = 100# float('inf')
    
        for epoch_idx in range(n_epochs):
            for step_idx in range(n_steps):
                inputs = next(gen)
                loss = self.train_step(inputs)
                
                v , _ , init_pos = inputs
                
                self.loss.append(loss)
                
                if stopping_criterion:
                    if step_idx > 5:
                        if loss >= np.mean(self.loss[-3:]): #STOPING CRITERION
                            print('Stoping criterion hit, ending training early.')
                            break
                    
                # Validation step
                val_loss = self.validation_step(test_inputs)
                self.val_loss.append(val_loss)
                
                if not self.trajectory_generator.use_pc:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_model_path = os.path.join(self.ckpt_dir, 'best_model.pth')
                        torch.save(self.model.state_dict(), best_model_path)
                        
                        if init_pos is not None:
                            # Vectorized position update
                            cumulative_v = torch.cumsum(v, dim=0)
                            visited_positions = init_pos.squeeze(1) + cumulative_v
                
                            place_cell_outputs = self.model.predict(inputs) # sequence_length x batch_size x Np
                
# =============================================================================
#                             # Vectorized standardization
#                             mean_vals = torch.mean(place_cell_outputs, dim=(0, 1))
#                             std_vals = torch.std(place_cell_outputs, dim=(0, 1))
#                             std_vals = torch.where(std_vals > 0, std_vals, torch.ones_like(std_vals))
#                             place_cell_outputs = 1 + (place_cell_outputs - mean_vals) / std_vals
#                             
#                             place_cell_outputs = torch.relu(place_cell_outputs)
# =============================================================================


                            # Min-Max Scaling
                            min_vals = torch.min(place_cell_outputs.view(-1), dim=0)[0]
                            max_vals = torch.max(place_cell_outputs.view(-1), dim=0)[0]
                
                            # Avoid division by zero by adding a small constant
                            denom = max_vals - min_vals + 1e-6
                            place_cell_outputs = (place_cell_outputs - min_vals) / denom



                            # self.model.place_cell_outputs = place_cell_outputs
                            # self.model.visited_positions = visited_positions

                

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')
                
                print('Epoch: {}/{}. Step {}/{}. Loss: {}. Val Loss: {}'.format(
                    epoch_idx, n_epochs, step_idx, n_steps,
                    np.round(loss, 2), np.round(val_loss, 2)))

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
# =============================================================================
#                 save_ratemaps(self.model, self.trajectory_generator,
#                               self.options, step=epoch_idx)
#                 
#                 
# =============================================================================

