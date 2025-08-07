"""
Authentication and key management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
import keyring
import json
from cryptography.fernet import Fernet
import base64
import os

router = APIRouter()

class StoreKeysRequest(BaseModel):
    exchange: str
    api_key: str
    api_secret: str

def get_encryption_key() -> bytes:
    """Get or create encryption key for storing secrets."""
    key_name = "wave_encryption_key"
    stored_key = keyring.get_password("wave", key_name)
    
    if not stored_key:
        # Generate new key
        key = Fernet.generate_key()
        encoded_key = base64.b64encode(key).decode()
        keyring.set_password("wave", key_name, encoded_key)
        return key
    else:
        return base64.b64decode(stored_key.encode())

def encrypt_secret(value: str) -> str:
    """Encrypt a secret value."""
    key = get_encryption_key()
    f = Fernet(key)
    encrypted = f.encrypt(value.encode())
    return base64.b64encode(encrypted).decode()

def decrypt_secret(encrypted_value: str) -> str:
    """Decrypt a secret value."""
    key = get_encryption_key()
    f = Fernet(key)
    encrypted_bytes = base64.b64decode(encrypted_value.encode())
    decrypted = f.decrypt(encrypted_bytes)
    return decrypted.decode()

@router.post("/keys")
async def store_keys(request: StoreKeysRequest):
    """Store encrypted exchange API keys."""
    try:
        # Encrypt the keys
        encrypted_key = encrypt_secret(request.api_key)
        encrypted_secret = encrypt_secret(request.api_secret)
        
        # Store in keyring (for now, later move to database)
        keyring.set_password("wave", f"{request.exchange}_api_key", encrypted_key)
        keyring.set_password("wave", f"{request.exchange}_api_secret", encrypted_secret)
        
        return {"status": "success", "message": "Keys stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store keys: {str(e)}")

@router.get("/keys/{exchange}")
async def get_keys(exchange: str):
    """Get stored keys for an exchange (for internal use)."""
    try:
        encrypted_key = keyring.get_password("wave", f"{exchange}_api_key")
        encrypted_secret = keyring.get_password("wave", f"{exchange}_api_secret")
        
        if not encrypted_key or not encrypted_secret:
            raise HTTPException(status_code=404, detail="Keys not found")
        
        # Decrypt keys (in practice, this would be done by internal services)
        api_key = decrypt_secret(encrypted_key)
        api_secret = decrypt_secret(encrypted_secret)
        
        return {
            "api_key": api_key,
            "api_secret": api_secret,
            "has_keys": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve keys: {str(e)}")

@router.delete("/keys/{exchange}")
async def delete_keys(exchange: str):
    """Delete stored keys for an exchange."""
    try:
        keyring.delete_password("wave", f"{exchange}_api_key")
        keyring.delete_password("wave", f"{exchange}_api_secret")
        return {"status": "success", "message": "Keys deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete keys: {str(e)}")

@router.get("/status")
async def auth_status():
    """Get authentication status."""
    exchanges = ["kraken"]  # Add more as needed
    status = {}
    
    for exchange in exchanges:
        key_exists = keyring.get_password("wave", f"{exchange}_api_key") is not None
        secret_exists = keyring.get_password("wave", f"{exchange}_api_secret") is not None
        status[exchange] = key_exists and secret_exists
    
    return {"exchanges": status}