#!/bin/bash

# Ensure .ssh directory exists
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Generate SSH key if it doesn't exist (completely non-interactive)
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -q
    echo "SSH key generated successfully"
else
    echo "SSH key already exists, skipping generation"
fi

cd ~/.ssh/ || exit

echo "Checking SSH connectivity to all clients..."

for ((CID=1;CID<=6;CID++)); do
    LOGIN=commnetpi0$CID@129.105.6.$((CID+16))
    echo "Checking client $CID ($LOGIN)..."
    
    # Test SSH connectivity without password
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $LOGIN exit 2>/dev/null; then
        echo "Client $CID: SSH already configured, skipping"
    else
        echo "Client $CID: Setting up SSH authorization"
        
        # Use sshpass for password authentication
        if sshpass -e scp -o StrictHostKeyChecking=no id_rsa.pub $LOGIN: 2>/dev/null; then
            if sshpass -e ssh -o StrictHostKeyChecking=no $LOGIN "mkdir -p .ssh && cat id_rsa.pub >> .ssh/authorized_keys && chmod 700 .ssh && chmod 600 .ssh/authorized_keys && rm id_rsa.pub" 2>/dev/null; then
                echo "Client $CID: SSH authorization successful"
            else
                echo "Client $CID: Failed to configure authorized_keys"
            fi
        else
            echo "Client $CID: Failed to copy public key"
        fi
    fi
done

echo "SSH authorization setup complete"