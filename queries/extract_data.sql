-- extract all products compatible with a given
SELECT DISTINCT O1.cod_modelo_color
FROM outfit_data O1, outfit_data O2
WHERE O1.cod_outfit = O2.cod_outfit
AND O2.cod_modelo_color = ?;




