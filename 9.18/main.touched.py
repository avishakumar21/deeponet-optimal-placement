#!/usr/bin/env python
# """
# Author: Minglang Yin, myin16@jhu.edu
# Solving Poisson Equation on Paramertized Domain. 

# Branch Net Input:       [u(x1), u(x2), ..., u(xn)]
# Trunk Net Input:        [x, y]
# Output:                 [u]
# """
import os
import torch
import time
import numpy as np
import scipy.io as io
import math

from utils import *
from opnn import *
from scipy.interpolate import Rbf
# from scipy.interpolate import interp2d
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    ## hyperparameters
    args = ParseArgument()
    epochs = args.epochs
    #device = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_step = args.save_step
    test_model = args.test_model
    if test_model == 1:
        load_path = "./CheckPts/model_chkpts.pt"

    PODMode = 10
    num_bc = 1 #204
    skip = 1
    dim_br1 = [PODMode*2, 100, 100, 100]
    dim_br2 = [num_bc, 150, 150, 150, 100] #150
    dim_tr = [2, 100, 100, 100]
    lmd = 0 #1e-6

    num_train = 400 ## assume 1000+500
    num_test = 59 ## note mesh coordinate in Laplace_test.mat save #900-909 
    num_cases = num_train + num_test
    
    ## create folders
    dump_train = './Predictions/Train/'
    dump_test = './Predictions/Test/'
    model_path = 'CheckPts/model_chkpts.pt'
    os.makedirs(dump_train, exist_ok=True)
    os.makedirs(dump_test, exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    
    ## dataset main
    datafile = "./Laplace_data.mat"
    dataset = io.loadmat(datafile)

    x = dataset["x_uni"]    #x: reference mesh                                   #!!!!!!!!!!!!!!!!
    x_mesh = dataset["x_mesh_data"]  #x_mesh: target mesh
    dx = x_mesh - x ## shape: num_case, num_points, num_dim
    u = dataset["u_data"]
    
    u_bc = dataset["u_bc"]
    u_bc = u_bc[:, ::skip]

    ## dataset supp
    #datafile_supp = "../Laplace_data_supp.mat"
    #dataset_supp = io.loadmat(datafile_supp)

    #x_mesh_supp = dataset_supp["x_mesh_data"]
    #dx_supp = x_mesh_supp - x ## shape: num_case, num_points, num_dim
    #u_supp = dataset_supp["u_data"]
    
    #u_bc_supp = dataset_supp["u_bc"]
    #u_bc_supp = u_bc_supp[:, ::skip]

    ## concat
    #x_mesh = np.concatenate((x_mesh, x_mesh_supp), axis=0)
    #dx = np.concatenate((dx, dx_supp), axis=0)
    #u = np.concatenate((u, u_supp), axis=0)
    #u_bc = np.concatenate((u_bc, u_bc_supp), axis=0)
    
    ## SVD
    dx_train = dx[:num_train]
    dx_test = dx[num_train:num_train+num_test]
    
    ## coefficient for training data
    dx1_train = dx_train[:, :, 0].transpose()
    dx2_train = dx_train[:, :, 1].transpose()
    Ux, coeff_x_train, Uy, coeff_y_train = SVD_reduce(dx1_train, dx2_train, PODMode)
    f_train = np.concatenate((coeff_x_train, coeff_y_train), axis=0).transpose()       #!!!!!!!!!!!!!
  
    ## coeffiient for testing data
    dx1_test = dx_test[:, :, 0].transpose()
    dx2_test = dx_test[:, :, 1].transpose()
    coeff_x_test = (Ux.transpose()).dot(dx1_test)
    coeff_y_test = (Uy.transpose()).dot(dx2_test)
    f_test = np.concatenate((coeff_x_test, coeff_y_test), axis=0).transpose()          #!!!!!!!!!!!!!

    ##########################
    u_train = u[:num_train]                                                            #!!!!!!!!!!!!!
    u_test = u[num_train:(num_train+num_test)]                                         #!!!!!!!!!!!!!
    f_bc_train = u_bc[:num_train, :]                                                   #!!!!!!!!!!!!!
    f_bc_test = u_bc[num_train:, :]                                                    #!!!!!!!!!!!!!

    ## tensor
    u_test_tensor = torch.tensor(u_test, dtype=torch.float).to(device)
    f_test_tensor = torch.tensor(f_test, dtype=torch.float).to(device)
    f_bc_test_tensor = torch.tensor(f_bc_test, dtype=torch.float).to(device)

    u_train_tensor = torch.tensor(u_train, dtype=torch.float).to(device)
    f_train_tensor = torch.tensor(f_train, dtype=torch.float).to(device)
    f_bc_train_tensor = torch.tensor(f_bc_train, dtype=torch.float).to(device)

    x_tensor = torch.tensor(x, dtype=torch.float).to(device)

    ## initialization
    model = opnn(dim_br1, dim_br2, dim_tr).to(device)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    if test_model == 0:
        train_loss = np.zeros((args.epochs, 1))
        test_loss = np.zeros((args.epochs, 1))
        rel_l2_err_his = np.zeros((args.epochs, 1))

        ## training
        def train(epoch, f, f_bc, x, y):
            model.train()
            def closure():
                optimizer.zero_grad()

                ## L2 regularization
                # regularization_loss = 0
                # for param in model._branch.parameters():
                #     regularization_loss += torch.sum(param**2)
                # for param in model._trunk.parameters():
                #     regularization_loss += torch.sum(param**2)
                #for param in model._out_layer.parameters():
                #    regularization_loss += torch.sum(param**2)

                loss = model.loss(f, f_bc, x, y)
                train_loss[epoch, 0] = loss

                loss.backward()
                return loss
            optimizer.step(closure)

        ## Iterations
        print('start training...', flush=True)
        tic = time.time()
        for epoch in range(0, epochs):
            if epoch == 10000:
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
            elif epoch == 90000:
                optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
            elif epoch == epochs - 1000:
                optimizer = torch.optim.LBFGS(model.parameters())

            ## Training
            train(epoch, f_train_tensor, f_bc_train_tensor, x_tensor, u_train_tensor)
            ## Testing
            loss_tmp = to_numpy(model.loss(f_test_tensor, f_bc_test_tensor, x_tensor, u_test_tensor))
            test_loss[epoch, 0] = loss_tmp

            ## Relative L2
            u_test_pred = to_numpy(model.forward(f_test_tensor, f_bc_test_tensor, x_tensor))
            rel_l2_err = (np.linalg.norm(u_test_pred - u_test, axis=1)/np.linalg.norm(u_test, axis=1)).mean()
            rel_l2_err_his[epoch, 0] = rel_l2_err

            ## testing error
            if epoch%100 == 0:
                # y_out = model.forward(f_test_tensor, x_tensor, u_test_tensor)
                # print(y_out[0])
                # print(y_enc[0])
                # print( ((y_out - y_enc)**2).mean()  )
                print(f'Epoch: {epoch}, Train Loss: {train_loss[epoch, 0]:.6f}, Test Loss: {test_loss[epoch, 0]:.6f}, Rel L2 test: {rel_l2_err_his[epoch, 0]}', flush=True)
                
            ## Save model
            if (epoch+1)%save_step == 0: 
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)

        toc = time.time()

        print(f'total training time: {int((toc-tic)/60)} min', flush=True)
        np.savetxt('./train_loss.txt', train_loss)
        np.savetxt('./test_loss.txt', test_loss)
        np.savetxt('./rel_l2_test.txt', rel_l2_err_his)

        ## plot loss function
        num_epoch = train_loss.shape[0]
        train_step = np.linspace(1, num_epoch, num_epoch)
        
        ## plot loss
        fig = plt.figure(constrained_layout=False, figsize=(6, 6))
        gs = fig.add_gridspec(1, 1)

        ax = fig.add_subplot(gs[0])
        ax.plot(train_step, train_loss.mean(axis=1), color='blue', label='Training Loss')
        ax.plot(train_step, test_loss.mean(axis=1), color='red', label='Test Loss', linestyle='dashed')
        ax.set_yscale('log')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.legend(loc='upper left')
        fig.savefig('./loss_his.png')

        ## save train
        u_train_pred = model.forward(f_train_tensor, f_bc_train_tensor, x_tensor)
        u_pred = to_numpy(u_train_pred)
        u_true = u_train

        num_plot = 201
        x_tmp = np.linspace(-2.5, 2.5, num_plot)
        xx, yy = np.meshgrid(x_tmp, x_tmp)
        grid_plot = np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None]), axis=1)

        for i in range(0, 10):
            u_pred_plot = u_pred[i]
            u_true_plot = u_true[i]
            x_coord = x_mesh[i]

            print(f"printing training case: {i}")

            ## figure
            fig = plt.figure(constrained_layout=False, figsize=(15, 5))
            gs = fig.add_gridspec(1, 3)

            ax = fig.add_subplot(gs[0])
            h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_true_plot)
            plt.colorbar(h)
            ax.set_aspect(1)
            ax.set_title("True data")

            ax = fig.add_subplot(gs[1])
            h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_pred_plot)
            plt.colorbar(h)
            ax.set_aspect(1)
            ax.set_title("prediction")

            ax = fig.add_subplot(gs[2])
            h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_true_plot - u_pred_plot)
            plt.colorbar(h)
            ax.set_aspect(1)
            ax.set_title("err")

            fig.savefig(dump_train + "case_" + str(i) + "_img.png")
            plt.close()

        
    else:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])


        
    ##############
    ## save test
    u_test_pred = model.forward(f_test_tensor, f_bc_test_tensor, x_tensor)
    u_pred = to_numpy(u_test_pred)
    u_true = u_test
    u_pred_save = []
    abs_err_pred_save = []

    ## MSE err
    mean_abs_err = (abs(u_pred - u_true)).mean(axis=1)

    rel_l2_err = np.linalg.norm(u_test - u_pred, axis=1)/np.linalg.norm(u_test, axis=1)
                        # np.mean()
    np.savetxt("mean_abs_err.txt", mean_abs_err)
    np.savetxt("rel_l2_err.txt", rel_l2_err)
    print(mean_abs_err.min())
    exit()

    exit()
    for i in range(0, 10):
        u_pred_plot = u_pred[i]
        u_true_plot = u_true[i]
        x_coord = x_mesh[i+num_train]
        print(f"printing testing case: {i}")

        ###
        fig = plt.figure(constrained_layout=False, figsize=(15, 5))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_true_plot)
        plt.colorbar(h)
        ax.set_aspect(1)
        ax.set_title("True data")

        ax = fig.add_subplot(gs[1])
        h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_pred_plot)
        plt.colorbar(h)
        ax.set_aspect(1)
        ax.set_title("prediction")

        ax = fig.add_subplot(gs[2])
        h = ax.scatter(x_coord[:, 0], x_coord[:, 1], c=u_true_plot - u_pred_plot)
        plt.colorbar(h)
        ax.set_aspect(1)
        ax.set_title("err")

        fig.savefig(dump_test + "case_" + str(i) + "_img.png")
        plt.close()

        u_pred_save.append(u_pred_plot)
        abs_err_pred_save.append(abs(u_pred_plot - u_true_plot))


    if test_model == 1:
        num2test = 400

        save_lib = {}
        save_lib["u_pred_canonical_plot"] = u_pred_save
        save_lib["x_canonical"] = x
        save_lib["x_mesh"] = x_mesh_supp[num2test:]
        save_lib["mesh_coords"] = dataset_supp["mesh_coords"][0][num2test:]
        save_lib["mesh_cells"] = dataset_supp["mesh_cells"][0][num2test:]
        save_lib["abs_err_pred_save"] = abs_err_pred_save
        #save_lib["mse"] = MSE_err
        #save_lib["rela_mse_plot"] = rel_MSE_err
        save_lib["u_pred"] = u_pred
        save_lib["u_true"] = u_true
        io.savemat("model_pred_vis.mat", save_lib)

if __name__ == "__main__":
    main()
