import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from database.models import Base, User, Category, Transaction

# Setup for in-memory SQLite database for testing
@pytest.fixture(scope="module")
def engine():
    return create_engine("sqlite:///:memory:")

@pytest.fixture(scope="module")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def session(engine, tables):
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()

def test_create_user(session):
    new_user = User(username="testuser", email="test@example.com", password_hash="hashedpassword")
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    assert new_user.user_id is not None
    assert new_user.username == "testuser"
    assert new_user.email == "test@example.com"
    assert new_user.password_hash == "hashedpassword"
    assert isinstance(new_user.created_at, datetime)
    assert isinstance(new_user.updated_at, datetime)
    assert new_user.created_at == new_user.updated_at # Initially, created and updated should be same

def test_create_category(session):
    user = User(username="catuser", email="cat@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    new_category = Category(user_id=user.user_id, category_name="Food", description="Groceries and Restaurants")
    session.add(new_category)
    session.commit()
    session.refresh(new_category)

    assert new_category.category_id is not None
    assert new_category.user_id == user.user_id
    assert new_category.category_name == "Food"
    assert new_category.description == "Groceries and Restaurants"
    assert new_category.owner.username == "catuser"

def test_create_transaction(session):
    user = User(username="transuser", email="trans@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    category = Category(user_id=user.user_id, category_name="Shopping")
    session.add(category)
    session.commit()
    session.refresh(category)

    new_transaction = Transaction(
        user_id=user.user_id,
        original_description="AMAZON.COM*ABCDE",
        cleaned_description="Amazon.com",
        amount=55.99,
        transaction_date=datetime(2023, 10, 26),
        category_id=category.category_id,
        categorization_confidence=0.95,
        needs_review=False,
        source_file_id="upload_123"
    )
    session.add(new_transaction)
    session.commit()
    session.refresh(new_transaction)

    assert new_transaction.transaction_id is not None
    assert new_transaction.user_id == user.user_id
    assert new_transaction.original_description == "AMAZON.COM*ABCDE"
    assert new_transaction.cleaned_description == "Amazon.com"
    assert new_transaction.amount == 55.99
    assert new_transaction.transaction_date == datetime(2023, 10, 26)
    assert new_transaction.category_id == category.category_id
    assert new_transaction.categorization_confidence == 0.95
    assert new_transaction.needs_review is False
    assert new_transaction.source_file_id == "upload_123"
    assert new_transaction.owner.username == "transuser"
    assert new_transaction.category.category_name == "Shopping"

def test_update_user(session):
    user = User(username="updateuser", email="update@example.com", password_hash="oldhash")
    session.add(user)
    session.commit()
    session.refresh(user)

    user.password_hash = "newhash"
    session.commit()
    session.refresh(user)

    assert user.password_hash == "newhash"
    assert user.updated_at > user.created_at

def test_delete_transaction(session):
    user = User(username="deluser", email="del@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    transaction = Transaction(
        user_id=user.user_id,
        original_description="Delete Me",
        amount=10.00,
        transaction_date=datetime.now()
    )
    session.add(transaction)
    session.commit()
    session.refresh(transaction)

    transaction_id = transaction.transaction_id
    session.delete(transaction)
    session.commit()

    deleted_transaction = session.query(Transaction).filter_by(transaction_id=transaction_id).first()
    assert deleted_transaction is None

def test_transaction_without_category(session):
    user = User(username="nocatuser", email="nocat@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    new_transaction = Transaction(
        user_id=user.user_id,
        original_description="Uncategorized Item",
        amount=100.00,
        transaction_date=datetime.now(),
        needs_review=True
    )
    session.add(new_transaction)
    session.commit()
    session.refresh(new_transaction)

    assert new_transaction.category_id is None
    assert new_transaction.needs_review is True

def test_user_category_relationship(session):
    user = User(username="reluser", email="rel@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    category1 = Category(user_id=user.user_id, category_name="Cat1")
    category2 = Category(user_id=user.user_id, category_name="Cat2")
    session.add_all([category1, category2])
    session.commit()
    session.refresh(category1)
    session.refresh(category2)

    assert len(user.categories) == 2
    assert category1 in user.categories
    assert category2 in user.categories

def test_user_transaction_relationship(session):
    user = User(username="reltransuser", email="reltrans@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    transaction1 = Transaction(user_id=user.user_id, original_description="T1", amount=10, transaction_date=datetime.now())
    transaction2 = Transaction(user_id=user.user_id, original_description="T2", amount=20, transaction_date=datetime.now() - timedelta(days=1))
    session.add_all([transaction1, transaction2])
    session.commit()
    session.refresh(transaction1)
    session.refresh(transaction2)

    assert len(user.transactions) == 2
    assert transaction1 in user.transactions
    assert transaction2 in user.transactions

def test_category_transaction_relationship(session):
    user = User(username="cattransuser", email="cattrans@example.com", password_hash="hash")
    session.add(user)
    session.commit()
    session.refresh(user)

    category = Category(user_id=user.user_id, category_name="Books")
    session.add(category)
    session.commit()
    session.refresh(category)

    transaction1 = Transaction(user_id=user.user_id, original_description="Book1", amount=15, transaction_date=datetime.now(), category_id=category.category_id)
    transaction2 = Transaction(user_id=user.user_id, original_description="Book2", amount=25, transaction_date=datetime.now() - timedelta(days=2), category_id=category.category_id)
    session.add_all([transaction1, transaction2])
    session.commit()
    session.refresh(transaction1)
    session.refresh(transaction2)

    assert len(category.transactions) == 2
    assert transaction1 in category.transactions
    assert transaction2 in category.transactions
