function caculatePrice() {
    var price = 0;
    var quantity = document.getElementById("quantity").value;
    var pricePerItem = document.getElementById("pricePerItem").value;
    price = quantity * pricePerItem;
    document.getElementById("price").value = price;
}
