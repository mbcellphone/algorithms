#!/usr/bin/env python3

"""
Module Name: tree_algorithms.py
Description: Examples for Tree Algorithms (Binary Search Tree, AVL Tree Insertion)
Author: Marvin Billings
Date: 12/02/2024
"""

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

# Binary Search Tree (BST) Insertion
class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, root, key):
        """
        Inserts a new key into the BST.
        :param root: The root of the BST.
        :param key: The key to insert.
        :return: The new root after insertion.
        """
        if root is None:
            return Node(key)
        if key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)
        return root

    def inorder(self, root):
        """
        Performs in-order traversal of the BST.
        :param root: The root of the BST.
        """
        if root:
            self.inorder(root.left)
            print(root.key, end=" ")
            self.inorder(root.right)

# AVL Tree Insertion
class AVLTree:
    def insert(self, root, key):
        """
        Inserts a new key into the AVL Tree.
        :param root: The root of the AVL Tree.
        :param key: The key to insert.
        :return: The new root after insertion.
        """
        if not root:
            return Node(key)
        if key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        root.height = 1 + max(self.get_height(root.left),
                              self.get_height(root.right))

        balance = self.get_balance(root)

        # Left Left Case
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)

        # Right Right Case
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)

        # Left Right Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # Right Left Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left),
                           self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left),
                           self.get_height(y.right))
        return y

    def right_rotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        z.height = 1 + max(self.get_height(z.left),
                           self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left),
                           self.get_height(y.right))
        return y

    def get_height(self, root):
        if not root:
            return 0
        return root.height

    def get_balance(self, root):
        if not root:
            return 0
        return self.get_height(root.left) - self.get_height(root.right)

    def inorder(self, root):
        """
        Performs in-order traversal of the AVL Tree.
        :param root: The root of the AVL Tree.
        """
        if root:
            self.inorder(root.left)
            print(root.key, end=" ")
            self.inorder(root.right)

# Main execution
if __name__ == "__main__":
    # Test Binary Search Tree (BST)
    print("Testing Binary Search Tree (BST) Insertion and In-Order Traversal")
    bst = BinarySearchTree()
    root_bst = None
    keys_bst = [20, 8, 22, 4, 12, 10, 14]
    for key in keys_bst:
        root_bst = bst.insert(root_bst, key)
    print("In-order traversal of BST:")
    bst.inorder(root_bst)
    print("\n")

    # Test AVL Tree
    print("Testing AVL Tree Insertion and In-Order Traversal")
    avl = AVLTree()
    root_avl = None
    keys_avl = [30, 20, 40, 10, 25, 50, 5]
    for key in keys_avl:
        root_avl = avl.insert(root_avl, key)
    print("In-order traversal of AVL Tree:")
    avl.inorder(root_avl)
    print()

