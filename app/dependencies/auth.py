# app/dependencies/auth.py

from jose import JWTError, jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from app.config import MONGODB_URI, JWT_SECRET_KEY, JWT_ALGORITHM
# from app.schemas.report_schemas import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_mongo_client():
    return AsyncIOMotorClient(MONGODB_URI)

# async def get_current_user(
#     token: str = Depends(oauth2_scheme),
#     mongo_client: AsyncIOMotorClient = Depends(get_mongo_client),
# ):
#     # 1) Decode the JWT
#     try:
#         payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
#         user_id: str = payload.get("id")
#         if not user_id:
#             raise HTTPException(status_code=401, detail="Invalid token payload")
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Could not validate credentials")

#     # 2) Retrieve the user document from MongoDB
#     db = mongo_client.get_default_database()
#     users_col = db["users"]
#     user_doc = await users_col.find_one({"_id": ObjectId(user_id)})
#     if not user_doc:
#         raise HTTPException(status_code=404, detail="User not found")

   
#     user_doc["id"] = str(user_doc["_id"])
  

#     return User(**user_doc)