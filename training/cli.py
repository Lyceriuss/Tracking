import os
import torch
from pathlib import Path
from train import train_model

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_float_input(prompt, default):
    while True:
        val = input(prompt)
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_int_input(prompt, default):
    while True:
        val = input(prompt)
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def list_checkpoints(checkpoint_dir="./checkpoints"):
    path = Path(checkpoint_dir)
    if not path.exists():
        return []
    return [f for f in path.glob("*.pth")]

def inspect_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(f"\n--- Checkpoint Info: {ckpt_path.name} ---")
    print(f"Epoch Reached : {checkpoint.get('epoch', 'Unknown')} / {checkpoint.get('total_epochs', 'Unknown')}")
    print(f"Best F1 Score : {checkpoint.get('best_f1', 0.0):.1f}%")
    print(f"Last LR       : {checkpoint.get('last_lr', 'Unknown')}")
    
    schema = checkpoint.get('schema', [])
    if schema:
        print(f"Labels ({len(schema)})  : {', '.join(schema[:4])} ...")
    print("-" * 40)

def main_menu():
    while True:
        clear_screen()
        print("=========================================")
        print("  Training pipeline ")
        print("=========================================")
        print("1. Start a New Training Run")
        print("2. Resume from Checkpoint")
        print("3. Inspect Checkpoint Info")
        print("4. Exit")
        print("=========================================")
        
        choice = input("Select an option (1-4): ")

        if choice == '1':
            run_name = input("\nEnter a name for this run: ")
            if not run_name.strip():
                continue
                
            epochs = get_int_input("Total Epochs [default 20]: ", 20)
            start_lr = get_float_input("Starting LR [default 0.1]: ", 0.1)
            end_lr = get_float_input("Ending LR (eta_min) [default 1e-5]: ", 1e-5)
            
            print("\n--- Configure p_mix Schedule ---")
            print("Valid options: 0.0, 0.2, 0.3, 0.4, 0.5")
            valid_pmix = [0.0, 0.2, 0.3, 0.4, 0.5]
            p_mix_schedule = []
            
            for i in range(epochs):
                while True:
                    val_str = input(f"Epoch {i+1} p_mix [default 0.0]: ")
                    val = float(val_str) if val_str else 0.0
                    if val in valid_pmix:
                        p_mix_schedule.append(val)
                        break
                    else:
                        print(f"Invalid! Must be one of: {valid_pmix}")
            
            train_model(
                run_name=run_name,
                start_lr=start_lr,
                end_lr=end_lr,
                total_epochs=epochs,
                p_mix_schedule=p_mix_schedule
            )
            input("\nTraining finished. Press Enter to return to menu.")

        elif choice == '2':
            ckpts = list_checkpoints()
            if not ckpts:
                input("\nNo checkpoints found. Press Enter to return.")
                continue
            
            print("\nAvailable Checkpoints:")
            for i, ckpt in enumerate(ckpts):
                print(f"[{i}] {ckpt.name}")
                
            idx = input(f"Select checkpoint to resume (0-{len(ckpts)-1}): ")
            try:
                selected_ckpt = ckpts[int(idx)]
                train_model(run_name=selected_ckpt.stem.replace("_best", ""), resume_checkpoint=selected_ckpt)
                input("\nTraining finished. Press Enter to return.")
            except (ValueError, IndexError):
                input("\nInvalid selection. Press Enter to return.")

        elif choice == '3':
            ckpts = list_checkpoints()
            if not ckpts:
                input("\nNo checkpoints found. Press Enter to return.")
                continue
                
            print("\nAvailable Checkpoints:")
            for i, ckpt in enumerate(ckpts):
                print(f"[{i}] {ckpt.name}")
                
            idx = input(f"Select checkpoint to inspect (0-{len(ckpts)-1}): ")
            try:
                inspect_checkpoint(ckpts[int(idx)])
                input("\nPress Enter to return to menu.")
            except (ValueError, IndexError):
                pass

        elif choice == '4':
            print("Exiting...")
            break

if __name__ == "__main__":
    main_menu()